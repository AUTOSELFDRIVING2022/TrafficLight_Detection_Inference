import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from torch import nn
import cv2
from source.modelTrafficLight import TrafficLightNet, TrafficLightNet_32x32, TrafficLightNet_32x32_noSTN, TrafficLightNet_64x64_noSTN, TrafficLightNet_64x64_coordConv, TrafficLightNet_64x32_noSTN, TrafficLightNet_64x32_coordConv
from source.model.modelTrafficLightLSTM import TrafficLightNet_64x32_LSTM, TrafficLightNet_128x128_LSTM
from source.model.resnet18LSTM import ResNetLSTM, BasicBlock
from source.model.TSM_model import TSN

class classifierTL(object):
    def __init__(self, classifierType='32x32', weightFile='None', batchSize = 1):
        checkpoint = torch.load(weightFile, map_location='cuda')
        if classifierType == '32x32':
            #self.modelTL = TrafficLightNet_32x32_noSTN(nclasses=17).to('cuda')
            #self.modelTL = TrafficLightNet_64x32_coordConv(nclasses=7).to('cuda')
            self.modelTL = TrafficLightNet_64x32_noSTN(nclasses=7).to('cuda')
        else:
            self.modelTL = TrafficLightNet_64x64_noSTN(nclasses=7).to('cuda')
            #self.modelTL = TrafficLightNet_64x64_coordConv(nclasses=7).to('cuda')
            #self.modelTL = TrafficLightNet_64x64_coordConv(nclasses=7).to('cuda')
        
        self.modelTL.load_state_dict(checkpoint)
        self.modelTL.eval()

        self.widthTLfar100m = 21
        self.heightTLfar100m = 7
        self.batchSize = batchSize

    def classify_tl_in_box(self, box,imageSrc, idx):
        width = imageSrc.shape[1]
        height = imageSrc.shape[0]
    
        x1 = 0 if int(box[0] * width) < 0 else int(box[0] * width)
        y1 = 0 if int(box[1] * height) < 0 else int(box[1] * height)
        x2 = 0 if int(box[2] * width) < 0 else int(box[2] * width)
        y2 = 0 if int(box[3] * height) < 0 else int(box[3] * height)

        if ((x2 - x1) < self.widthTLfar100m) and ((y2 - y1) < self.heightTLfar100m):
            print("Traffic light bbox dim error")
            return -1

        orgBoxImg = cv2.cvtColor(imageSrc[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)
        boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(64,32)).transpose(2,0,1), axis=0))
        #boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(32,32)).transpose(2,0,1), axis=0))
        #boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(64,64)).transpose(2,0,1), axis=0))
        x = (boxImg / 255.0).to('cuda')

        # orgBoxImg = cv2.cvtColor(imageSrc[y1:y2,x1:x2],cv2.COLOR_BGR2LAB)
        # boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(64,32)).transpose(2,0,1), axis=0))
        # #boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(32,32)).transpose(2,0,1), axis=0))
        # #boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(64,64)).transpose(2,0,1), axis=0))
        # x = (boxImg / 255.0).to('cuda')

        #Infer
        outputTL = self.modelTL(x)
        pred = torch.argmax(outputTL, dim=1).cpu().numpy()
    
        debug = False
        if debug:
            cv2.imwrite("./data/{}_pred_{}.jpg".format(idx,pred[0]),cv2.cvtColor(orgBoxImg,cv2.COLOR_BGR2RGB))
            #cv2.imwrite("./data/{}_pred_{}.jpg".format(idx,pred[0]),cv2.cvtColor(orgBoxImg,cv2.COLOR_LAB2BGR))
        return pred[0]

    def tl_classification(self, bboxes, imageSrc, fname = ''):
        for box_idx, box in enumerate(bboxes):
            tl_class = self.classify_tl_in_box(box,imageSrc, fname)
            box.append(tl_class)
        return bboxes
    
class classifierTL_Temporal(object):
    def __init__(self, classifierType='resnetLSTM', weightFile='None', batchSize = 1, nclass = 7, num_frames = 10):
        checkpoint = torch.load(weightFile, map_location='cuda')
        if classifierType == 'conv_64x32':
            self.modelTL = TrafficLightNet_64x32_LSTM(nclasses=nclass).to('cuda')  
        elif classifierType == 'conv_128x128':
            self.modelTL = TrafficLightNet_128x128_LSTM(nclasses=nclass).to('cuda')  
        elif classifierType == 'resnetLSTM':
            self.modelTL = ResNetLSTM(BasicBlock, [2, 2, 2, 2], num_classes = nclass).to('cuda')
        elif classifierType == 'TSN':
            self.modelTL = TSN(num_class = nclass, num_segments = num_frames, modality = 'RGB', base_model='resnet18', is_shift = False).to('cuda')
        elif classifierType == 'TSM':
            self.modelTL = TSN(num_class = nclass, num_segments = num_frames, modality = 'RGB', base_model='resnet18', is_shift = True).to('cuda')
        
        
        self.modelTL.load_state_dict(checkpoint)
        self.modelTL.eval()
        self.modelTL.to('cuda')

        self.widthTLfar100m = 21
        self.heightTLfar100m = 7
        self.batchSize = batchSize

    def classify_tl_in_box(self, boxes,imageSrc, idx):
        width = imageSrc.shape[1]
        height = imageSrc.shape[0]
        
        boxImg = []
        for box in boxes:
    
            x1 = 0 if int(box[0] * width) < 0 else int(box[0] * width)
            y1 = 0 if int(box[1] * height) < 0 else int(box[1] * height)
            x2 = 0 if int(box[2] * width) < 0 else int(box[2] * width)
            y2 = 0 if int(box[3] * height) < 0 else int(box[3] * height)

            if box[0] == -1 and box[1] == -1 and box[2] == -1 and box[3] == -1:
                pass
            elif ((x2 - x1) < self.widthTLfar100m) and ((y2 - y1) < self.heightTLfar100m):
                print("Traffic light bbox dim error")
                return -1
            else:
                orgBoxImg = cv2.cvtColor(imageSrc[y1:y2,x1:x2],cv2.COLOR_BGR2RGB) / 255.0
                orgBoxImg = cv2.resize(orgBoxImg,(64,32)).transpose(2,0,1)
                boxImg.append(orgBoxImg)
        
        _boxImg = torch.from_numpy(np.expand_dims(boxImg,axis=0))
        _boxImg = _boxImg.type(torch.FloatTensor).to('cuda')
        #temporalBox = torch.stack(_boxImg).to('device')
        #frames = video.permute(0,1,2,3)
        
        #boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(64,32)).transpose(2,0,1), axis=0))
        #boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(32,32)).transpose(2,0,1), axis=0))
        #boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(64,64)).transpose(2,0,1), axis=0))
             
        
        
        # orgBoxImg = cv2.cvtColor(imageSrc[y1:y2,x1:x2],cv2.COLOR_BGR2LAB)
        # boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(64,32)).transpose(2,0,1), axis=0))
        # #boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(32,32)).transpose(2,0,1), axis=0))
        # #boxImg = torch.from_numpy(np.expand_dims(cv2.resize(orgBoxImg,(64,64)).transpose(2,0,1), axis=0))
        # x = (boxImg / 255.0).to('cuda')

        #Infer
        outputTL = self.modelTL(_boxImg)
        pred = torch.argmax(outputTL, dim=1).cpu().numpy()
    
        debug = False
        if debug:
            cv2.imwrite("./data/{}_pred_{}.jpg".format(idx,pred[0]),cv2.cvtColor(orgBoxImg,cv2.COLOR_BGR2RGB))
            #cv2.imwrite("./data/{}_pred_{}.jpg".format(idx,pred[0]),cv2.cvtColor(orgBoxImg,cv2.COLOR_LAB2BGR))
        return pred[0]

    def make_temporal_data(self, bboxes, imageSrc):
        n = len(bboxes)
        #bboxes = list(bboxes)
        import statistics
        tl_n_median = int(np.median([len(bboxes) for bboxes in bboxes]))
        for idx, bbox in enumerate(bboxes):
            if len(bbox) == tl_n_median:
                pass
            elif len(bbox) < tl_n_median:
                while len(bboxes[idx]) < tl_n_median:
                    print("\n Warning TL bbox is not detected well: dummy data added")
                    #bboxes[idx].append([])
                    bboxes[idx] = np.concatenate(([[-1,-1,-1,-1,-1,-1,-1,-1,-1]],bboxes[idx]), axis=0)
                    #bbox = np.concatenate((bbox, [[-1,-1,-1,-1,-1,-1,-1,-1,-1]]), axis=0)
            elif len(bbox) > tl_n_median:
                while len(bboxes[idx]) > tl_n_median:
                    print("\n Warning TL bbox is not detected well: data removed")
                    #bboxes[idx].pop(0)
                    bboxes[idx] = np.delete(bboxes[idx],1,0)
                
        #bboxes.pop(1)
        _bboxes = np.swapaxes(bboxes,0,1)
        return _bboxes
    
    def tl_classification(self, bboxes, imageSrc, fname = ''):
        ### Convert axis:
        # Temporal_Size x TL_Number x BBOX -> TL_Number x Temporal_Size x BBOX
        # TBD: need to improve, very rough method --> have to improve  
        bboxes = self.make_temporal_data(bboxes, imageSrc)
        bboxes_classes = []
        for box_idx, box in enumerate(bboxes):
            tl_class = self.classify_tl_in_box(box,imageSrc[box_idx], fname)
            bboxes_classes.append(tl_class)
        return bboxes, bboxes_classes
                
