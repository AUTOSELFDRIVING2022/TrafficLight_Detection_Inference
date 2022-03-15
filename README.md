# TrafficLight Classification module with YoloV4 detector and SORT tracker.

###### base code of YoloV4 
A minimal PyTorch implementation of YOLOv4.
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet

# 0. Weight Download
## 0.1 darknet
- Drive (Link: https://drive.google.com/file/d/1wKXAmAHSzWsiTFqEkjTuQqoFiQwbclSF/view?usp=sharing)
- Save pretrained weight to this path: ./data/weight/
# 1. Inference 
- Load pretrained darknet modle and darknet weights to do the inference
- Inference Traffic Light Classification:
'''sh
python inference_YOLOv4_TL_Classification_TRACK.py --det_weight {weight_file_of_detector} --det_cfg {config_file_of_detector} --cl_weight {classification_weight} --input {input_file_directory} --showResult {save_result_true_false}    
