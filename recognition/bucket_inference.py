import numpy as np
import cv2

try:
    from . import config, utils
    config.Config({})

except (SystemError, ImportError) as e:
    import utils
    import config

ARGS = config.ARGS

bucket_specs = [0, 256, 512, 896]

def get_bucket_id(width):
    index = [i for i in range(len(bucket_specs) -1) if width >=bucket_specs[i] and width < bucket_specs[i+1]]
    if len(index) == 0:
        index = len(bucket_specs) - 2
    else:
        index = index[0]
    return index

def make_batch(images):

    bucket_data = [{"inputs_numpy_array":[],"seq_len_numpy_array":[],"sequence":[]} for i in range(len(bucket_specs)-1)]

    for i,img in enumerate(images):
        im = img
        h,w = im.shape
        pad = 0.05
        im = utils.pad(im,[int(pad*h/2),int(pad*h/2),int(3*pad*h/2),int(3*pad*h/2)])
        im = utils.fix_height(im,ARGS.image_height)
        width_img = im.shape[1]
        bucket_id = get_bucket_id(width_img)
        bucket_width = bucket_specs[bucket_id+1]
        if im.shape[1] > bucket_width:
            # print("what is this")
            im = cv2.resize(im,(bucket_width,ARGS.image_height))
            width_img = im.shape[1]
        else:
            im = utils.pad(im,[0,0,0,bucket_width - im.shape[1]],cv2.BORDER_REPLICATE)

        quant_width = config.expand_for_ctc(width_img)
        bucket_data[bucket_id]["seq_len_numpy_array"].append(config.size_after_convolution(quant_width))
        bucket_data[bucket_id]["inputs_numpy_array"].append(im)
        bucket_data[bucket_id]["sequence"].append(i)
    all_sequences = []
    batch_size = ARGS.val_batch_size
    for bucket_id in range(len(bucket_specs)-1):
        allready_present = len(bucket_data[bucket_id]["sequence"])
        if  allready_present== 0:
            continue
        if allready_present % batch_size != 0:
            toadd = ((allready_present // batch_size) + 1)*batch_size - allready_present
            for i_toadd in range(toadd):
                bucket_data[bucket_id]["inputs_numpy_array"].append(bucket_data[bucket_id]["inputs_numpy_array"][-1])
                bucket_data[bucket_id]["seq_len_numpy_array"].append(bucket_data[bucket_id]["seq_len_numpy_array"][-1])
                bucket_data[bucket_id]["sequence"].append(-1)
        bucket_data[bucket_id]["inputs_numpy_array"] = np.asarray(bucket_data[bucket_id]["inputs_numpy_array"],dtype=np.float32)[:,:,:,np.newaxis]
        bucket_data[bucket_id]["seq_len_numpy_array"] = np.asarray(bucket_data[bucket_id]["seq_len_numpy_array"],dtype=np.int32)
        all_sequences.extend(bucket_data[bucket_id]["sequence"])
    return bucket_data, all_sequences
