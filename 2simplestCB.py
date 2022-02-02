import cv2
import math
import numpy as np
import os
import imageio

def simplest_cb(img, percent=1):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)

if __name__ == '__main__':
    IMG_PATH = 'cropped_face_dir/acne_red.jpg'
    img = cv2.imread(IMG_PATH)
    UPLOAD_FOLDER = "cropped_face_dir"
    IMG_next_name = os.path.splitext(os.path.basename(IMG_PATH))[0] + '_CB.jpg'
    # img = cv2.resize(img, (500,500))
    out = simplest_cb(img, 1)
    # out2 = cv2.cvtColor(out, cv2.COLOR_BGR2Lab)
    cv2.imshow("Before", img)
    cv2.imshow("After", out)
    out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
    filename = os.path.join(UPLOAD_FOLDER, IMG_next_name)
    imageio.imwrite(filename, out)
    # cv2.imshow("after", out2)
    cv2.waitKey(0)