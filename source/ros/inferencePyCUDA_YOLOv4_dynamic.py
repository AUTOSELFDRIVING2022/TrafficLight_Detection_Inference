import sys
import os
import time
import argparse
import numpy as np
import cv2
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from source.utils import *
from trt_wrapper_common import TRTInference

def main(enginePath, imgPath, imgSize, inTestType, saveResult, nClasses, batch_size=1):
    if nClasses == 20:
        classNamesfile = '../data/voc.names'
    elif nClasses == 80:
        classNamesfile = '../data/coco.names'
    else:
        classNamesfile = '../data/KETIDB.names'

    classNames = load_class_names(classNamesfile)
    
    trtWrapper = TRTInference(enginePath, batch_size, multiThread=False)

    
    for i in range(2):  # This 'for' loop is for speed check    # Because the first iteration is usually longer
        boxes, elapsed_time, imageSrc = detect_yolo_trt(trtWrapper, imgPath, imgSize, nClasses, batch_size)
        
    if saveResult:
        outDir = "../pred/"
        fileName = [outDir + os.path.splitext(os.path.basename((img)))[0] + "_detected.jpg" for img in imgPath]
        
        for b in range(batch_size):
            plot_boxes_cv2(imageSrc[b], boxes[b], savename=fileName[b], class_names=classNames)

    print("Total Frame Rate = %.2f fps" %(batch_size/elapsed_time))

def preprocess(image_path, max_size=(512,512), batch_size=3):
    if batch_size > 1:
        ori_imgs = [cv2.imread(img_path) for img_path in image_path]
        imgs = [cv2.resize(img, (max_size[0], max_size[1])) for img in ori_imgs]
        norm = [img / 255.0 for img in imgs]

        imgsT = [img.transpose(2, 0, 1).astype(np.float32) for img in norm]
    else:
        ori_imgs = cv2.imread(image_path[0]) 
        imgs = cv2.resize(ori_imgs, (max_size[0], max_size[1]))
        norm = imgs / 255.0

        imgsT = norm.transpose(2, 0, 1).astype(np.float32)

    return ori_imgs, imgsT

def detect_yolo_trt(trtWrapper, image_src, image_size, num_classes, batch_size=1):
    IN_IMAGE_H, IN_IMAGE_W = image_size
    print(IN_IMAGE_H, IN_IMAGE_W)
    # Input
    imageSrc, imgs = preprocess(image_src, image_size, batch_size)

    ## Inference Start
    start_time = time.time()
    trt_outputs = trtWrapper.infer(imgs)

    print('Len of outputs: ', len(trt_outputs))
    trt_outputs[0] = trt_outputs[0].reshape(batch_size, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(batch_size, -1, num_classes)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    ##Post-processing START
    start_time = time.time()
    print("shape of trt",len(trt_outputs))
    #for b in range(batch_size):
    boxes = post_processing(0.4, 0.6, trt_outputs)
    print("BOXES:",boxes)
    end_time = time.time()
    elapsed_time_post = end_time - start_time

    print("Inference time --> Inference time:{0:3.5f}({1:3.5f}fps), Postprocess time:{2:3.5f}({3:3.5f}fps)".format(
        elapsed_time, batch_size/elapsed_time, elapsed_time_post, batch_size/elapsed_time_post))

    return boxes, elapsed_time, imageSrc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("--engine", nargs="?", type=str, default="../bin/yolov4_1_3_608_608_static_fp16.engine", help="engine file name",)
    parser.add_argument("--width", nargs="?", type=int, default=608, help="target input width",)
    parser.add_argument("--height", nargs="?", type=int, default=608, help="target input height", )
    #parser.add_argument("--input", nargs="?", type=str, default="../data/dog.jpg", help="input file name or input directory depends on inputTestType", )
    parser.add_argument("--input", nargs="*", type=str, default="./test/img1.png,./test/img2.jpg,./test/img3.jpg, ./test/img4.jpg, ./test/img5.jpg", help="input file name or input directory depends on inputTestType", )
    parser.add_argument("--inputTestType", nargs="?", type=str, default="single", help="input test TYPE: single / dir", )
    parser.add_argument("--showResult", nargs="?", type=bool, default=True, help="Show segmentation result, result file saved in /pred/ folder",)
    parser.add_argument("--classes", nargs="?", type=int, default=80, help="weight file name",)
    parser.add_argument("--batchSize", nargs="?", type=int, default=3, help="batch size")

    args = parser.parse_args()
    print(args.input)

    saveResult = args.showResult
    enginePath = args.engine
    imgPath = args.input
    nClasses = args.classes
    inTestType = args.inputTestType
    imgSize = (args.width, args.height)
    batch_size = args.batchSize

    main(enginePath, imgPath, imgSize, inTestType, saveResult, nClasses, batch_size)
