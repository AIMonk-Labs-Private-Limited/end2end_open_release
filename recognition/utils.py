import cv2

def pad(im,pad = [0,0,0,0],type_border = cv2.BORDER_REPLICATE):
    top_pad = pad[0]
    bot_pad = pad[1]
    left_pad = pad[2]
    right_pad = pad[3]
    img_bw=cv2.copyMakeBorder(im, top=top_pad, bottom=bot_pad, left=left_pad, right=right_pad, borderType= type_border)
    return img_bw

def fix_height(im,height):
    h,w = im.shape[0],im.shape[1]
    width = int(float(height)*w/h)
    im = cv2.resize(im,(width,height))
    return im
