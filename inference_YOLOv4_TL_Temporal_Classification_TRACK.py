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

#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from source.utils import *
from tqdm import tqdm

from source.tl_classification import classifierTL_Temporal
from source.tl_tracker import trackerTL
from source.darknet2pytorch import Darknet

working_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

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

    return [ori_imgs, imgsT]

def main(detectorWeightPath, imgPath, imgSize, inTestType, saveResult, nClasses, detectorCfgPath, classificationWeightPath, 
         outResultPath, classNamesPath, tempDataSize):
    if nClasses == 20:
        classNamesfile = classNamesPath.format(working_dir)
    elif nClasses == 80:
        classNamesfile = classNamesPath.format(working_dir)
    else:
        classNamesfile = classNamesPath.format(working_dir)
    
    classNames = load_class_names(classNamesfile)
    batchSize = 1

    #Classification
    modeTL = True 
    classificationTL = True
    trackingTL = True
    if modeTL: 
        #Traffic Light Classification Mode.
        if classificationTL:
            #weightFile = "./data/weight/trafficLight_model_64_32_best_rgb_0.946_extra.pt"
            #weightFile = "./data/weight/trafficLight_model_64_32_best_lab_coordconv_0.964_extra.pt"

            #recognizeTL = classifierTL(classifierType='64x64', weightFile=weightFile, batchSize=batchSize)
            recognizeTL = classifierTL_Temporal(classifierType='resnetLSTM', weightFile=classificationWeightPath, batchSize=batchSize)
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
        imgPaths = []
        imgFiles = []
        for imgFile in tqdm(sorted(os.listdir(imgPath)), desc='YoloV4', mininterval=0.01):
            _, ext = os.path.splitext(os.path.basename((imgFile)))
        
            if ext not in [".png", ".jpg"]:
                continue
            
            if len(imgPaths) < tempDataSize:
                imgPaths.append(imgPath+imgFile)
                imgFiles.append(imgFile)
                doInf = False
            elif len(imgPaths) == tempDataSize:
                doInf = True
            else: 
                doInf = False
                pass
            
            
            if doInf:
                boxes = []
                elapsed_time = [] 
                imageSrc = []

                ### Read input images in DIR
                #img_path = [os.path.join(imgPath, imgFile) for i in range(tempDataSize)]

                for _imgPath in imgPaths:
                    _boxes, _elapsed_time, _imageSrc = detect_yolo_trt(model, _imgPath, imgSize, nClasses)
                    boxes.append(_boxes[0])
                    elapsed_time.append(_elapsed_time)
                    imageSrc.append(_imageSrc)
                
                
            
            new_boxes = []
            if modeTL and doInf:
                tl_box = []
                tl_box_id = []
                front_camera_idx = 0 if batchSize == 1 else 1
                traffic_light_class_number = classNames.index("Traffic Light")
                outDir = outResultPath
                fname_classification = outDir + imgFile[:-4] + "_cropp.jpg"
                fname_tracking = outDir + imgFile[:-4] + "_cropp.jpg"

                for batch_idx, bboxes in enumerate(boxes):
                    _tl_box = []
                    _tl_box_id = []
                    _new_boxes = []
                    for box_idx, box in enumerate(bboxes):
                        if box[5] == traffic_light_class_number:
                            if trackingTL:
                                box.append(0)
                            _tl_box.append(box)
                            _tl_box_id.append(box_idx)
                        else:
                            _new_boxes.append(box)
                    tl_box.append(_tl_box)
                    tl_box_id.append(_tl_box_id)
                    new_boxes.append(_new_boxes)
                
                if trackingTL:
                    outDir = outResultPath
                    fileNameTrack = outDir + imgFile[:-4] + "_tracked.jpg"
                    tl_box_tr = []
                    #if len(tl_box) > 0:
                    for idx, _tl_box in enumerate(tl_box):
                        _tl_box_tr, imageSrcTrack = tracking.track(_tl_box, imageSrc[idx], fname_tracking)
                        tl_box_tr.append(_tl_box_tr)
                
                if classificationTL:
                    #tl_boxes_classified = tl_classification(boxes, imageSrc, 1, modelTL, traffic_light_class_number,fname_classification)
                    fileName = imgFile + "_detected.jpg"

                    tl_box, tl_box_classes = recognizeTL.tl_classification(tl_box_tr, imageSrc, fname=fileName)
                    
                    #if __DEBUG_USER__:
                    #    outDir = "./pred/ros/"
                    #    fileName = imgFile + "_detected.jpg"
                    #    plot_boxes_cv2_tl(imageSrc, tl_box, savename=fileName, class_names=classNames)

                totalTime += sum(elapsed_time)
                processImg += tempDataSize

                if saveResult:
                    outDir = outResultPath
                    
                    for tl_idx, _tl_box in enumerate(tl_box):
                        for idx in range(tempDataSize):
                            fileName_corrected = outDir + imgFiles[idx][:-4] + "_detected2.jpg"
                            imageSrc[idx] = plot_boxes_cv2_tl_temporal(imageSrc[idx], _tl_box[idx], tl_box_classes[tl_idx], savename=fileName_corrected, class_names=classNames)
                
                ##Remove buff in image path
                imgPaths = []
                imgFiles = []
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
    conf_thresh_tl = 0.4
    
    ##Traffic Light idx
    idx_tl = 6
    
    nms_thresh = 0.6
    IN_IMAGE_H, IN_IMAGE_W = image_size

    imgSrc0, imgIn0 = preprocess(image_src, image_size)
    #img_in = np.expand_dims(img_in, axis=0)
    #img_in = np.ascontiguousarray(img_in)
    #print("Shape of the network input: ", img_in.shape)
    #print(img_in)

    #print('Length of inputs: ', len(inputs))
    #inputs[0].host = img_in
    img = torch.from_numpy(imgIn0).to('cuda')
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
    
    return boxes, elapsed_time, imgSrc0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("--det_weight", nargs="?", type=str, default="./data/weight/yolov4_211203.weights", help="Detector's pretrained weight",)
    parser.add_argument("--det_cfg", nargs="?", type=str, default="./data/yolov4.cfg", help="Detector's cfg (YoloV4 config file))",)
    parser.add_argument("--cl_weight", nargs="?", type=str, default="./data/weight/trafficLight_model.pt", help="Detector's pretrained weight",)
    parser.add_argument("--width", nargs="?", type=int, default=608, help="target input width",)
    parser.add_argument("--height", nargs="?", type=int, default=608, help="target input height", )
    parser.add_argument("--input", nargs="?", type=str, default="./data/tl_complex_test/", help="input file name or input directory depends on inputTestType", )
    parser.add_argument("--inputTestType", nargs="?", type=str, default="dir", help="input test TYPE: single / dir", )
    parser.add_argument("--showResult", nargs="?", type=bool, default=True, help="Show segmentation result, result file saved in /pred/ folder",)
    parser.add_argument("--resultPath", nargs="?", type=str, default="./pred/ros_temporal/", help="Result path",)
    parser.add_argument("--classNamesPath", nargs="?", type=str, default="./data/KETIDB.names", help="Detector Class Names",)
    parser.add_argument("--classes", nargs="?", type=int, default=7, help="weight file name",)
    parser.add_argument("--temp_data_size", nargs="?", type=int, default=10, help="temporal window size",)

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
    tempDataSize = args.temp_data_size
    
    main(detectorWeightPath, imgPath, imgSize, inTestType, saveResult, nClasses, detectorCfgPath, classificationWeightPath, outResultPath, classNamesPath, tempDataSize)
