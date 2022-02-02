import os
import cv2
from PIL import ImageStat, Image
import math

def pass_or_fail_through_file_size(file_path):
    if os.path.isfile(file_path):
        n= os.path.getsize(file_path) # Byte
        n = n //1024 # KB
        if( n >= 1000): # more than 1KB
            print("pass {0}KB file".format(n))
            return 1
        else:
            print("Fail because of {0}KB".format(n))
            return 0
    else:
        print("There is no file")
        return 0

# https://www.analyticsvidhya.com/blog/2020/09/how-to-perform-blur-detection-using-opencv-in-python/
def detect_blurry(file_path):
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()

    if fm > 200:
        print("Blurry : pass {0} file".format(fm))
        return 1
    else:
        print("Blurry : Fail because of {0}".format(fm))
        return 0


def brightness( im_file ):
   im = Image.open(im_file)
   stat = ImageStat.Stat(im) #calculate histogram
   band = stat.mean # average value
   r = band[0]
   g = band[1]
   b = band[2]
   score = math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
   if score > 120:
        print("Brightness : pass {0} file".format(score))
        return 1
   else:
        print("Brightness : fail {0} file".format(score))
        return 0

def calculate_face_per_image(file_path):
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width, _ = image.shape
    img_size = height * width
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05,
	    minNeighbors=5, minSize=(30, 30),
	    flags=cv2.CASCADE_SCALE_IMAGE)
    for (_,_,w,h) in faces:
        face_size = w * h
    score = round(face_size / img_size, 4)
    if score >= 0.4:
        print("FPI : pass {0} file".format(score))
        return 1
    else:
        print("fPI : fail {0} file".format(score))
        return 0

def pass_or_fail(img_file):
    sc1 = pass_or_fail_through_file_size(img_file)
    sc2 = detect_blurry(img_file)
    sc3 = brightness(img_file)
    sc4 = calculate_face_per_image(img_file)

    whole_score = sc1 + sc2+ sc3 + sc4
    if whole_score == 4:
        return print("pass")
    else:
        return print("fail")

if __name__ == '__main__':
    file_path = 'data/galaxynote.jpg'
    # pass_or_fail_through_file_size(file_path)
    # detect_blurry(file_path)
    # print(brightness(file_path))
    # print(calculate_face_per_image(file_path))
    pass_or_fail(file_path)