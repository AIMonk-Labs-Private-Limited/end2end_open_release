import cv2, random
import numpy as np

def expand_box(pts, factor):
    (tl, tr, br, bl) = pts[0], pts[1], pts[2], pts[3]
    factorx = 2.0*np.linalg.norm(tl-bl)*factor/np.linalg.norm(tl-tr)

    ##expanding height first
    tl -= (bl-tl)*factor/2
    tr -= (br - tr) * factor / 2
    br += (br - tr) * factor / 2
    bl += (bl - tl) * factor / 2

    ##expanding width then
    tl -= (tr-tl)*factorx/2
    tr += (tr - tl) * factorx / 2
    br += (br - bl) * factorx / 2
    bl -= (br - bl) * factorx / 2

def get_transformed_text_dims(pts):
    # top_left, top_right, bottom_right and bottom_left text vertices.
    factor = random.random()*0.2
    expand_box(pts,factor)
    (tl, tr, br, bl) = pts[0], pts[1], pts[2], pts[3]

    topWidth = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    bottomWidth = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    newWidth = int( (topWidth + bottomWidth) / 2 ) #L2 distance

    leftHeight = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    rightHeight = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    newHeight = int( (leftHeight + rightHeight) / 2 ) #L2 distance

    return newWidth, newHeight

def perform_perspective_transform(img, pts):
    # calculate the dimensions of the transformed img
    newWidth, newHeight = get_transformed_text_dims(pts)
    dst = np.array([[0, 0], [newWidth - 1, 0], [newWidth - 1, newHeight - 1], [0, newHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts[[0,1,2,3]], dst)
    warped = cv2.warpPerspective(img, M, (newWidth, newHeight))
    return warped

def crop_words_in_image(img, boxes):
    crop_img_list = []
    for bbox in boxes:
        try:
            croppedImg = perform_perspective_transform(img, np.array(bbox, dtype = "float32"))
            crop_img_list.append(croppedImg)
        except:
            crop_img_list.append([])
    return crop_img_list
