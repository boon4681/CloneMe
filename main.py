from PIL import Image, ImageGrab, ImageFont, ImageDraw
import numpy as np
import cv2
import time

a_th = cv2.THRESH_OTSU | cv2.THRESH_BINARY
cap = cv2.VideoCapture(0)

def overlay(background, overlay, x, y):
    w = background.shape[1]
    h = background.shape[0]
    if x >= w or y >= h:
        return background
    h, w = overlay.shape[0], overlay.shape[1]
    if x + w > w:
        w = w - x
        overlay = overlay[:, :w]
    if y + h > h:
        h = h - y
        overlay = overlay[:h]
    if overlay.shape[2] < 4:
        overlay = np.concatenate([overlay, np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255], axis=2)
    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    if(y < 0 or x < 0):
        ax,ay = abs(x),abs(y)
        masks = mask[ay:h, ax:w]
        background[0:h+y, 0:w+x] = (1.0 - masks) * \
            background[0:h+y, 0:w+x] + masks * overlay_image[ay:h, ax:w]
    else:
        background[y:y+h, x:x+w] = (1.0 - mask) * \
            background[y:y+h, x:x+w] + mask * overlay_image
    return background


def do(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img, 0, 255, a_th)
    binary = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
    H = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=5)
    V = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=5)
    cross = cv2.bitwise_or(H, V)
    cnts = cv2.findContours(cross, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(cross, cnts, [255,255,255])
    im_np = cv2.bitwise_and(image,image,mask=cross)
    b, g, r = cv2.split(im_np)
    rgba = [b,g,r, cross]
    im_np = cv2.merge(rgba,4)
    res = overlay(image,im_np,-200,0)
    res = overlay(res,im_np,200,0)
    res = overlay(res,im_np,0,0)
    return res

def main():
    ret, image = cap.read()
    cv2.imshow("CloneMe",do(image))
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()

while True:
    main()