#encoding=utf-8

import cv2
import sys
import numpy as np
import os.path
from glob import glob

def detect(filename, cascade_file="haarcascade_frontalface_alt.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" %  cascade_file)
        
    cascade = cv2.CascadeClassifier(cascade_file)
    
    #image = cv2.imread(filename)
    image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8),-1)
    if image == None:
        return
        
    if image.shape[2]==1:
        return
        
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(96, 96))
        
    for i,(x,y,w,h) in enumerate(faces):
        print("x:{} y:{} w:{} h:{}".format(x,y,w,h))
        face = image[y: y + h, x:x + w, :]
        face = cv2.resize(face, (96,96))
        save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        cv2.imwrite("face/" + save_filename, face)
    #print("===========End ========")    
if __name__ == '__main__':
    if os.path.exists('face') is False:
        os.makedirs('face')
    path='D:\\python\\workspace\\DL\\alert\\Crawler\\MM\\佐佐木明希\\*.jpg'
    file_list = glob(path)
    print("file_list:",len(file_list))
    for filename in file_list:
        print(filename)
        detect(filename)