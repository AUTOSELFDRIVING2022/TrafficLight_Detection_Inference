import sys
import os
import time
import argparse

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import torch

from source.utils import *
from tqdm import tqdm

from source.tl_classification import classifierTL
from source.tl_tracker import trackerTL
from source.darknet2pytorch import Darknet

working_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TRT_LOGGER = trt.Logger()

def read_img(img_path, size):
    #print("Read Input Image from : {}".format(img_path))
    img         = cv2.imread(img_path)
    img_resized = cv2.resize(img, (size[1], size[0]))  # uint8 with RGB mode
    img         = img_resized.astype(np.float32)
    img         = img / 255.0

    # NHWC -> NCHW
    img         = img.transpose(2, 0, 1)
    img         = np.ascontiguousarray(img)
    img         = img[np.newaxis,:]

    images      = img.astype(np.float32)
    return images

def preprocess(image_path, max_size=(512,512)):
    
    ori_imgs = cv2.imread(image_path)
    imgsRe = cv2.resize(ori_imgs, (max_size[0], max_size[1]))
    norm = imgsRe / 255.0

    imgsT = norm.transpose(2, 0, 1).astype(np.float32)

    return ori_imgs, imgsT

def main(detectorWeightPath, imgPath, imgSize, inTestType, saveResult, nClasses, detectorCfgPath, classificationWeightPath, outResultPath, classNamesPath, classification_with_detection):
    if nClasses == 20:
        classNamesfile = classNamesPath.format(working_dir)
    elif nClasses == 80:
        classNamesfile = classNamesPath.format(working_dir)
    else:
        classNamesfile = classNamesPath.format(working_dir)
    
    classNames = load_class_names(classNamesfile)
    batchSize = 1

    #Classification
    modeTL = classification_with_detection 
    classificationTL = True
    trackingTL = True
    if modeTL: 
        #Traffic Light Classification Mode.
        if classificationTL:
            #recognizeTL = classifierTL(classifierType='64x64', weightFile=weightFile, batchSize=batchSize)
            #recognizeTL = classifierTL(classifierType='32x32', weightFile=classificationWeightPath, batchSize=batchSize)
            recognizeTL = classifierTL(classifierType='resnet', weightFile=classificationWeightPath, batchSize=batchSize)

        #Tracking 
        if modeTL and trackingTL:
            tracking = trackerTL(trackerType='SORT', classNames=classNames, batchSize=batchSize)

    ### YoloV4 model initialize
    yolov4_cfg = detectorCfgPath
    yolov4_weight = detectorWeightPath
    
    model = Darknet(yolov4_cfg).to('cuda')
    #model.print_network()
    model.load_weights(yolov4_weight)
    model.to('cuda').eval()
    
    IN_IMAGE_H, IN_IMAGE_W = imgSize
    
    #context.set_binding_shape(0, (1, 3, IN_IMAGE_H, IN_IMAGE_W))
    if inTestType == "dir":
        processImg = 0
        totalTime = 0

        #print("---------------")
        #print(imgPath)

        for imgFile in tqdm(sorted(os.listdir(imgPath)), desc='YoloV4', mininterval=0.01):
            _, ext = os.path.splitext(os.path.basename((imgFile)))
        
            if ext not in [".png", ".jpg"]:
                continue
            
            ### Read input images in DIR
            img_path = os.path.join(imgPath, imgFile)

            boxes, elapsed_time, imageSrc = detect_yolo_trt(model, img_path, imgSize, nClasses)
            
            new_boxes = []
            if modeTL:
                tl_box = []
                tl_box_id = []
                front_camera_idx = 0 if batchSize == 1 else 1
                traffic_light_class_number = classNames.index("TL")
                outDir = outResultPath
                fname_classification = outDir + imgFile[:-4] + "_cropp.jpg"
                fname_tracking = outDir + imgFile[:-4] + "_cropp.jpg"

                for batch_idx, bboxes in enumerate(boxes):
                    for box_idx, box in enumerate(bboxes):
                        if box[5] == traffic_light_class_number and batch_idx == front_camera_idx:
                            tl_box.append(box)
                            tl_box_id.append(box_idx)
                        else:
                            new_boxes.append(box)
                        
                if classificationTL:
                    #tl_boxes_classified = tl_classification(boxes, imageSrc, 1, modelTL, traffic_light_class_number,fname_classification)
                    fileName = imgFile + "_detected.jpg"

                    tl_box = recognizeTL.tl_classification(tl_box, imageSrc, fname=fileName)
                    
                    #if __DEBUG_USER__:
                    #    outDir = "./pred/ros/"
                    #    fileName = imgFile + "_detected.jpg"
                    #    plot_boxes_cv2_tl(imageSrc, tl_box, savename=fileName, class_names=classNames)

                if trackingTL:
                    outDir = outResultPath
                    fileNameTrack = outDir + imgFile[:-4] + "_tracked.jpg"
                    
                    #if len(tl_box) > 0:
                    tl_box_tr, imageSrcTrack = tracking.track(tl_box, imageSrc, fname_tracking)
                    
                    #if __DEBUG_USER__:
                    #    cv2.imwrite(fileNameTrack, imageSrcTrack) 

                totalTime += elapsed_time
                processImg += 1

                if saveResult:
                    outDir = outResultPath
                    fileName_corrected = outDir + imgFile + "_detected2.jpg"
                    plot_boxes_cv2_tl(imageSrc, tl_box_tr, savename=fileName_corrected, class_names=classNames)
            else: 
                totalTime += elapsed_time
                processImg += 1

                if saveResult:
                    outDir = outResultPath
                    fileName_corrected = outDir + imgFile + "_detected2.jpg"
                    plot_boxes_cv2(imageSrc, boxes[0], savename=fileName_corrected, class_names=classNames)
        print("Total Frame Rate = %.2f fps" %(processImg/totalTime))
    else:
        #imageSrc = cv2.imread(imgPath)
        for i in range(2):  # This 'for' loop is for speed check    # Because the first iteration is usually longer
            boxes, elapsed_time, imageSrc = detect_yolo_trt(model, img_path, imgSize, nClasses)

        if saveResult:
            outDir = outResultPath
            fileName = outDir + (os.path.splitext(os.path.basename(imgPath))[0]) + "_detected.jpg"
            #print("File Name:",fileName)
            plot_boxes_cv2(imageSrc, boxes[0], savename=fileName, class_names=classNames)

        print("Total Frame Rate = %.2f fps" %(1/elapsed_time))

def detect_yolo_trt(model, image_src, image_size, num_classes):
    conf_thresh = 0.4
    conf_thresh_tl = 0.35
    
    ##Traffic Light idx
    idx_tl = 6
    
    nms_thresh = 0.6
    IN_IMAGE_H, IN_IMAGE_W = image_size

    imageSrc, img_in = preprocess(image_src, image_size)
    #img_in = np.expand_dims(img_in, axis=0)
    #img_in = np.ascontiguousarray(img_in)
    #print("Shape of the network input: ", img_in.shape)
    #print(img_in)

    #print('Length of inputs: ', len(inputs))
    #inputs[0].host = img_in
    img = torch.from_numpy(img_in).to('cuda')
    if img.ndimension() == 3:
            img = img.unsqueeze(0)
    
    start_time = time.time()
    trt_outputs = model(img)

    #print('Len of outputs: ', len(trt_outputs))
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    start_time = time.time()
    
    boxes = post_processing(conf_thresh, conf_thresh_tl, idx_tl, nms_thresh, trt_outputs)
    
    end_time = time.time()
    elapsed_time_post = end_time - start_time

    #print("Inference time --> Inference time:{0:3.5f}({1:3.5f}fps), Postprocess time:{2:3.5f}({3:3.5f}fps)".format(
    #    elapsed_time, 1/elapsed_time, elapsed_time_post, 1/elapsed_time_post))
    
    return boxes, elapsed_time, imageSrc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("--det_weight", nargs="?", type=str, default="./data/weight/yolov4_211203.weights", help="Detector's pretrained weight",)
    parser.add_argument("--det_cfg", nargs="?", type=str, default="./data/yolov4.cfg", help="Detector's cfg (YoloV4 config file))",)
    #parser.add_argument("--cl_weight", nargs="?", type=str, default="./data/weight/trafficLight_model_64_32_best_rgb_0.946_extra.pt", help="Detector's pretrained weight",)
    parser.add_argument("--cl_weight", nargs="?", type=str, default="./data/weight/best_epoch_resnet18_220701.bin", help="Detector's pretrained weight",)
    parser.add_argument("--width", nargs="?", type=int, default=608, help="target input width",)
    parser.add_argument("--height", nargs="?", type=int, default=608, help="target input height", )
    #parser.add_argument("--input", nargs="?", type=str, default="./data/tl_complex_test/", help="input file name or input directory depends on inputTestType", )
    parser.add_argument("--input", nargs="?", type=str, default="./data/Sony_New_Camera_0630/Camera0/", help="input file name or input directory depends on inputTestType", )
    parser.add_argument("--inputTestType", nargs="?", type=str, default="dir", help="input test TYPE: single / dir", )
    parser.add_argument("--showResult", nargs="?", type=bool, default=True, help="Show segmentation result, result file saved in /pred/ folder",)
    #parser.add_argument("--resultPath", nargs="?", type=str, default="./pred/ros/", help="Result path",)
    #parser.add_argument("--resultPath", nargs="?", type=str, default="./pred/Sony_New_Camera_0630/Camera1/", help="Result path",)
    parser.add_argument("--resultPath", nargs="?", type=str, default="./pred/Sony_New_Camera_0630/Camera0_NEW_CLASSIFIER/", help="Result path",)
    parser.add_argument("--classNamesPath", nargs="?", type=str, default="./data/KETIDB.names", help="Detector Class Names",)
    parser.add_argument("--classes", nargs="?", type=int, default=7, help="weight file name",)
    parser.add_argument("--tl_classification", type=bool, default=True, help="Run YoloV4 with Traffic Light Classification.",)

    args = parser.parse_args()

    saveResult = args.showResult
    detectorWeightPath = args.det_weight
    detectorCfgPath = args.det_cfg
    classificationWeightPath = args.cl_weight
    
    imgPath = args.input
    nClasses = args.classes
    inTestType = args.inputTestType
    imgSize = (args.width, args.height)
    outResultPath = args.resultPath
    classNamesPath = args.classNamesPath
    classification_with_detection = args.tl_classification
    
    main(detectorWeightPath, imgPath, imgSize, inTestType, saveResult, nClasses, detectorCfgPath, classificationWeightPath, outResultPath, classNamesPath, classification_with_detection)
