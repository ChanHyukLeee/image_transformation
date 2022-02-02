import cv2 as cv
import math
import numpy as np
import os
from os import listdir
from os.path import join, isfile, splitext
import imageio

def read_img(img_file):
    img = cv.imread(img_file)
    return img

def cb(img):
    Red = []
    Green = []
    Blue = []
    blank = np.zeros(img.shape[:2], dtype="uint8")
    print(blank.shape)
    
    Blue, Red, Green = cv.split(img)

    R_avg = np.mean(Red)
    G_avg = np.mean(Green)
    B_avg = np.mean(Blue)
    
    R_inv = 1 / R_avg
    G_inv = 1 / G_avg
    B_inv = 1 / B_avg
    
    M = max(R_inv, G_inv, B_inv)

    R_scale = (R_inv / M) * Red
    G_scale = (G_inv / M) * Green
    B_scale = (B_inv / M) * Blue

    R_scale = R_scale.astype('uint8')
    G_scale = G_scale.astype('uint8')
    B_scale = B_scale.astype('uint8')
    
    CB = cv.merge([B_scale, R_scale, G_scale])
    return CB

if __name__ == "__main__":
    IMG_PATH = 'cropped_face_dir/acne_red.jpg'
    UPLOAD_FOLDER = "cropped_face_dir"
    IMG_next_name = os.path.splitext(os.path.basename(IMG_PATH))[0] + '_3CB.jpg'
    img = cv.imread(IMG_PATH)
    img_cb = cb(img)
    cv.imshow('before', img)
    cv.imshow("after", img_cb)
    out = cv.cvtColor(img_cb,cv.COLOR_BGR2RGB)
    filename = os.path.join(UPLOAD_FOLDER, IMG_next_name)
    imageio.imwrite(filename, out)
    cv.waitKey(0)