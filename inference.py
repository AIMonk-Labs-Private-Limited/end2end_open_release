import os
import gc
import time
import argparse
import torch
import cv2
import numpy as np
from crops_generation import crop_words_in_image
from recognition import run_inference
from detection import (TextSpotter, time_synchronized, 
                    normalizeMeanVariance, resize_aspect_ratio, get_boxes_cpu, 
                    adjustResultCoordinates, get_files, 
                    clean_checkpoint, loadImage)
import shutil
from typing import List, Dict, Any

torch.set_grad_enabled(False)
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='E2E OCR detection')
parser.add_argument('--detection_model', type=str,required=True,
                    help='pretrained detector model path')
parser.add_argument('--recognition_model', type=str, required=True, 
                    help='pretrained ocr model path')
parser.add_argument('--input_dir', required=True, type=str,
                    help='folder path to input images')
parser.add_argument('--output_dir', default="./result",
                    help="path where output images and results should be stored")
parser.add_argument('--gpu_id', type=int, required=True, 
                    help='gpu to use for inerence.')
parser.add_argument('--show_time', default=False, action='store_true',
                    help='show processing time')
parser.add_argument('--region_threshold', default=0.35, type=float,
                    help='Region map threshold')
parser.add_argument('--link_threshold', default=0.3, type=float,
                    help='Affinity map threshold')
parser.add_argument('--symbol_threshold', default=0.2, type=float,
                    help='Symbol map threshold')
parser.add_argument('--dilate_with_symbol', default=False, action="store_true",
                    help="Consider symbol map during final dilation in postprocessing")
parser.add_argument('--neglect_direction', default=False, action="store_true",
                    help="Neglect direction map in postprocessing even if it's predicted")
parser.add_argument('--neglect_symbolmap', default=False, action="store_true",
                    help="Neglect symbol map in postprocessing even if it's predicted")
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda for inference')
parser.add_argument('--float16', default=False, action="store_true",
                    help="Whether to run inference with float16 inference")
parser.add_argument('--canvas_size', default=1280, type=int,
                    help='Max image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float,
                    help='image magnification ratio')
parser.add_argument('--fuse_modules', default=False, action="store_true",
                    help="Whether to fuse Conv, BN and Relu layers before inference")

def test_net(net: TextSpotter, image: np.ndarray, config: Dict[str, Any], 
             link_threshold: float, region_threshold: float, 
             neglect_direction: bool, neglect_symbolmap: bool,
             symbol_threshold: float, dilate_with_symbol: bool, 
             cuda: bool, image_name: str = "default") -> List[np.ndarray]: 
    """Function to run the inference given image array, network and other info
    
    Args:
        net: End to End network using which inference should be run
        image: input image to run the inference on
        config: config dictionary containing various info required for inference
        link_threshold: Threshold for binarising affinity map
        region_threshold: Threshold for binarising region map
        neglect_direction: Whether to neglect the direction map if it's predicted
        neglect_symbolmap: Whether to neglect the symbol map if it's predicted
        symbol_threshold: Threshold for binarising symbol map
        dilate_with_symbol: Whether to consider symbol map also in dilation step
        cuda: Whether to use gpu or not
        gpu_decode: Whether to do the postprocessing in GPU or not
        image_name: Name of the input image
    """
    detector_features = None
    available_maps = config['model_params']['detector_params']['heatmap_classes'].copy()
    
    assert net.do_detection, "test_net function cannot work without detection module"
    t0 = time_synchronized()
    # Preprocessing the input image
    device = "cuda" if cuda else "cpu"
    # Changing the aspect ratio of the input image to required shape
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, args.canvas_size, 
                                                                 interpolation="bilinear", 
                                                                 mag_ratio=args.mag_ratio, 
                                                                 device=device)
    ratio_h = ratio_w = 1 / target_ratio
    # performing normalizing step
    x = normalizeMeanVariance(img_resized)
    if args.float16: x = x.half()
    t1 = time_synchronized()
    # Calling the forward functions of the detector model on preprocessed image(x)
    backbone_features = net.forward_basenet(x)
    detector_features = net.forward_detector(backbone_features)
    ret_heatmap = detector_features[0].float() if (detector_features[0] is not None) else None
    ret_word_direction = detector_features[1].float() if (detector_features[1] is not None) else None
    if neglect_direction: ret_word_direction = None
    if neglect_symbolmap:
        ret_heatmap = ret_heatmap[:, 0:2, :, :]
        if 'symbol' in available_maps: available_maps.remove('symbol')
        if 'uppercase' in available_maps: available_maps.remove('uppercase')
    t2 = time_synchronized()
    # Postprocessing where we get the boxes. 
    if 'uppercase' in available_maps: ret_heatmap = ret_heatmap[:, :-1, : , :]
    # import pdb;pdb.set_trace()
    sort_boxes, _, _ = get_boxes_cpu(ret_heatmap, ret_word_direction, 
                                      link_threshold, region_threshold,
                                      symbol_threshold, dilate_with_symbol)
    t3 = time_synchronized()
    # Removing offset and scale of input tensor from box locations
    boxes = adjustResultCoordinates(sort_boxes, ratio_w, ratio_h)
    # Show the time stats for the inference runned on single image
    if args.show_time:
        print("Detection Inference time stats on /{}:".format(image_name))
        print("Input shape, Height {} Width {}".format(x.shape[2], x.shape[3]))
        print("Preprocessing Image : {:.4f}".format(t1 - t0))
        print("TextDet forward {:.4f}, Posprocessing(ConnectComp) {:.4f}".format(t2 - t1, t3 - t2))
    return boxes

def test_ocr(run_handler: run_inference.Run, image: np.ndarray, 
             boxes: List, result_folder: str, image_name: str):
    """Function to run the inference given image array, network and other info
    
    Args:
        run_handler: OCR inference model to run
        image: input image to run inference on
        boxes: predicted bounding boxes from End to End detector
        result_folder: the path to directory where to save the crops along with predicted text
        image_name: name of the image on which inference runs
    """
    t4 = time.time() 
    # As preprocessing to run inference, we have to make crops 
    # from predicted boxes and the respective input image.
    images = crop_words_in_image(image, boxes)
    images = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    t5 = time.time()
    # Running the inference with the OCR model on crops.
    all_preds = []
    for image in images:
        preds = run_handler.infer_bucketize([image])
        all_preds.extend(preds)
    t6 = time.time()
    output_ocr_dir = os.path.join(result_folder, os.path.splitext(os.path.basename(image_path))[0])
    if os.path.isdir(output_ocr_dir): shutil.rmtree(output_ocr_dir)
    os.mkdir(output_ocr_dir)
    print("Crops saving to {}".format(output_ocr_dir))
    # For visualizing the crops along with predicted word transcript
    # This is the way we are saving the crops and respective predicted text
    for id, (crop, pred_text) in enumerate(zip(images, all_preds)):
        if crop.any():
            save_path = output_ocr_dir+"/"+pred_text+".jpg"
            cv2.imwrite(save_path, crop)
        else:
            print("skipping crop num:{}".format(id))
    print("Number of crops saved {}".format(len(images)))
    t7 = time.time()
    # Show the time stats for the inference runned on single image
    if args.show_time:
        print("OCR Inference time stats on /{}:".format(image_name))
        print("Preprocessing Image:{:.4f}".format(t5 - t4))
        print("Running Inference on all crops:{:.4f}".format(t6 - t5))
        print("Saving the crops along with predicted text:{:.4f}".format(t7 - t6))

if __name__ == '__main__':
    
    args = parser.parse_args()
    if os.path.isdir(args.input_dir):
        image_list, _, _ = get_files(args.input_dir)
    else:
        image_list = [args.input_dir]
    result_folder = args.output_dir
    
    if os.path.isdir(result_folder): shutil.rmtree(result_folder)
    os.makedirs(result_folder)

    try:
        # First we are running the detection on the images
        # This will output the predicted boxes. 
        if args.cuda: args.cuda = torch.cuda.is_available()
        model_state_dict = torch.load(args.detection_model, 
                                      map_location = "cpu" if not args.cuda else "cuda")
        config = model_state_dict['config']
        config['model_params']['do_ocr'] = False
        # Calling the detection model and then loading the pretrained model
        net = TextSpotter(config)
        print('Loading weights from checkpoint (' + args.detection_model + ')')
        model_state_dict = model_state_dict['ema'] if \
                            ('ema' in model_state_dict and (model_state_dict['ema'] is not None)) \
                            else model_state_dict['model']
        restore_result = net.load_state_dict(clean_checkpoint(model_state_dict, net), strict=False)
        if len(restore_result.missing_keys) > 1:
            raise ValueError("Some keys aren't restored from the check point")
        if args.cuda: net = net.cuda()
        if args.fuse_modules: net.fuse_modules()
        net.eval()
        if args.float16: net = net.half()
        t = time.time()
        # Doing the detection inference on images
        all_boxes = []
        for k, image_path in enumerate(image_list):
            image = loadImage(image_path)
            print("Test Image {}/{}".format(k, len(image_list)))
            boxes = test_net(net, image, config, 
                             args.link_threshold, args.region_threshold,
                             args.neglect_direction,
                             args.neglect_symbolmap,args.symbol_threshold, 
                             args.dilate_with_symbol, args.cuda,  
                             image_name=os.path.basename(image_path))
            all_boxes.append(boxes)
        # Removing and freeing space, since once we completed on detection
        del net, model_state_dict
        gc.collect()
        if args.cuda:
            torch.cuda.empty_cache()
        # Calling the OCR model and loading the pretrained model
        run_handler = run_inference.Run(gpu_to_use=args.gpu_id, model_dir=args.recognition_model)
        # Then doing the OCR inference on the images
        for k, image_path in enumerate(image_list):
            image = loadImage(image_path)
            print("Test Image {}/{}".format(k, len(image_list)))
            test_ocr(run_handler=run_handler, image=image, boxes=all_boxes[k], 
                     result_folder=result_folder, 
                     image_name=os.path.basename(image_path))
    
    except KeyboardInterrupt:
        print("Keyboard interrupted")
    print("elapsed time : {}s".format(time.time() - t))
