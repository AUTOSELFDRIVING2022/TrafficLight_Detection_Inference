# TrafficLight Classification module with YoloV7 detector
This source is only includes inference code.
Our detector has a seven different classes:
 - Person, Bike, Car, Bus, Truck, Traffic Sign, Traffic Light, Emergency Car
Our traffic light classification supports a 12 traffic light types.
 - green, green_left, off, other, pedestrain, red, red_left, red_yellow, yellow, yellow21, yellow_green4, yellow_other (South Korea).
###### base code of YoloV7 
A PyTorch implementation of YOLOv7.
- Paper Yolov7: https://arxiv.org/abs/2207.02696
- Source code: https://github.com/WongKinYiu/yolov7

###### TrafficLight Classification trainer code
Our traffic light classification code (Link: https://github.com/AUTOSELFDRIVING2022/TrafficLight_Classification_Trainer)

# 0. Weight Download
- YoloV7 weight Drive(Link: https://drive.google.com/file/d/1_bop55xJF-W0p3l7S2RS6NTLUxqcfU3V/view?usp=sharing)
- Resnet34 Traffic Light Classification weight Drive(Link: https://drive.google.com/file/d/1EA_MMeS59hREHRm72qTVdeMOPVqjgfGI/view?usp=sharing)
- Save pretrained weight to this path: ./data/weight/

# 1. Inference 
- Load pretrained yolov7 modle weights.
- Inference Traffic Light Classification:
```sh
python inference_YOLOv4_TL_Classification_TRACK.py --det_weight {weight_file_of_detector} --det_cfg {config_file_of_detector} --cl_weight {classification_weight} --input {input_file_directory} --showResult {save_result_true_false} 
```   
