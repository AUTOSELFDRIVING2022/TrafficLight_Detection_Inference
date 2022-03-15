import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from torch import nn
import cv2
from source.utils_deep_sort.draw import draw_boxes
from source.deep_sort import build_tracker
from source.sort import Sort as SORT

class trackerTL(object):
    def __init__(self, trackerType='SORT', classNames='None', batchSize = 1):
        self.traffic_light_class_number = classNames.index("Traffic Light")
        
        self.front_camera_idx = 0 if batchSize == 1 else 1
        self.trackerType = trackerType

        if self.trackerType == 'SORT':
            self.tracking = SORT(iou_threshold=0.005, min_hits=0, max_age=30)
        elif self.trackerType == 'DeepSORT':
            #deepSORT
            deepSORT = './source/deep_sort/deep/checkpoint/ckpt.t7'
            self.tracking = build_tracker(deepSORT,use_cuda=True)
    def boxScale(self, boxes):
        new_box = []
        if boxes: 
            for box in boxes:
                temp = []
                x1 = 0 if int(box[0] * self.width) < 0 else int(box[0] * self.width)
                y1 = 0 if int(box[1] * self.height) < 0 else int(box[1] * self.height)
                x2 = 0 if int(box[2] * self.width) < 0 else int(box[2] * self.width)
                y2 = 0 if int(box[3] * self.height) < 0 else int(box[3] * self.height)
                new_box.append([x1, y1, x2, y2, box[4],box[5],box[6]])
                #temp.append(x1) 
                #temp.append(y1)
                #temp.append(x2)
                #temp.append(y2)
                #temp.append(box[4])
                #new_box.append(temp)
        else:
            new_box=np.empty((0, 5))
        return np.array(new_box).astype(np.float32)
    
    def boxDeScale(self, boxes):
        new_box = []
        for box in boxes:
            temp = []
            x1 = 0 if (box[0] / self.width) < 0 else (box[0] / self.width)
            y1 = 0 if (box[1] / self.height) < 0 else (box[1] / self.height)
            x2 = 0 if (box[2] / self.width) < 0 else (box[2] / self.width)
            y2 = 0 if (box[3] / self.height) < 0 else (box[3] / self.height)
            new_box.append([x1, y1, x2, y2, box[4],box[5],box[6],box[7],box[8]])
            #temp.append(x1) 
            #temp.append(y1)
            #temp.append(x2)
            #temp.append(y2)
            #temp.append(box[4])
            #new_box.append(temp)
        
        return np.array(new_box).astype(np.float32)
    
    def track(self, boxes_batch, imageSrc, fname):
        self.imageSrc = imageSrc
        self.width = self.imageSrc.shape[1]
        self.height = self.imageSrc.shape[0]
        self.fname = fname
        self.boxes = self.boxScale(boxes_batch)

        if self.trackerType == 'SORT':
            return self.trackSORT()
        elif self.trackerType == 'DeepSORT':
            return self.trackDeepSORT()

    def trackSORT(self):
        tracked_box = self.tracking.update(self.boxes)
        ###tracked_box 
        
        #print(tracked_box)
        if len(tracked_box) != len(self.boxes):
            print("miss boxes")

        ##For debug
        #_bbox_xyxy = self.boxes[:, :4]
        #self.imageSrc = self.draw_boxesDebug(self.imageSrc, _bbox_xyxy)
        
        # draw boxes for visualization
        if len(tracked_box) > 0:
            bbox_xyxy = tracked_box[:, :4]
            identities = tracked_box[:, 7]
            self.imageSrc = draw_boxes(self.imageSrc, bbox_xyxy, identities)
        else: ##For debug
            self.imageSrc = self.imageSrc

        return self.boxDeScale(tracked_box), self.imageSrc
    
    def draw_boxesDebug(self, img, bbox, identities=None, offset=(-5,5)):
    #def draw_boxesDebug(self, img, bbox, identities=None, offset=(0,0)):
        for i,box in enumerate(bbox):
            x1,y1,x2,y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[1]
            y1 += offset[0]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0    
            color = [255,0,0]
            label = '{}{:d}'.format("", id)
            cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
            ###Put ID text
            #cv2.putText(img,label,(x1,y1), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,255,255], 1)


            #-----lines
            #center = ((round((x1+x2)/2),round((y1+y2)/2)))
            #pts[id].append(center)
            #cv2.circle(img,(round((x1+x2)/2),round((y1+y2)/2)),1,color,5)
            # for j in range(1, len(pts[id])):
            #     if pts[id][j-1] is None or pts[id][j] is None:
            #         continue
            #     thickness = int(np.sqrt(64/float(j+1)) * 2)
            #     cv2.line(img,(pts[id][j-1]),(pts[id][j]), color, thickness)
        return img

    def trackDeepSORT():
        if trackingTL:
                width = 1920
                height = 1080
                results = []
                bbox_xywh = []
                cls_conf = []
                cls_ids = []
                for box in boxes[0]:
                    bbox_xywh.append(box[0:5])
                    cls_conf.append(box[5]) ##BUG
                    cls_ids.append(box[6])##BUG
                #start_tracking = time.time()
                
                if cls_conf is not None:
                    #-----copy
                    list_fin = []
                    for box in bbox_xywh:
                        temp = []
                        x1 = 0 if int(box[0] * width) < 0 else int(box[0] * width)
                        y1 = 0 if int(box[1] * height) < 0 else int(box[1] * height)
                        x2 = 0 if int(box[2] * width) < 0 else int(box[2] * width)
                        y2 = 0 if int(box[3] * height) < 0 else int(box[3] * height)

                        temp.append(x1) 
                        temp.append(y1)
                        temp.append(x2)
                        temp.append(y2)
                        temp.append(box[4])
                        list_fin.append(temp)
                    new_bbox = np.array(list_fin).astype(np.float32)

                    #-----#-----mask processing filter the useless part
                    mask = [6]# keep specific classes the indexes are corresponded to coco_classes
                    mask_filter = []
                    for i in cls_ids:
                        if i in mask:
                            mask_filter.append(1)
                        else:
                            mask_filter.append(0)
                    new_cls_conf = []
                    new_new_bbox = []
                    new_cls_ids = []
                    new_sort_bbox = []
                    for i in range(len(mask_filter)):
                        if mask_filter[i]==1:
                            # tmp = []
                            # tmp.append(new_bbox[i])
                            # tmp.append(cls_conf[i])
                            # new_sort_bbox.append(tmp)
                            new_new_bbox.append(new_bbox[i])
                            new_cls_conf.append(cls_conf[i])
                            new_cls_ids.append(cls_ids[i])
                    new_bbox =  np.array(new_new_bbox).astype(np.float32)
                    #new_bbox =  np.array(new_sort_bbox).astype(np.float32)
                    cls_conf =  np.array(new_cls_conf).astype(np.float32)
                    cls_ids  =  np.array(new_cls_ids).astype(np.float32) 
                    #-----#-----

                    # do tracking
                    #outputs = deepsort.update(new_bbox, cls_conf, imageSrc)
                    outputs = tracking.update(new_bbox)

                    # draw boxes for visualization
                    imageSrcTrack = imageSrc
                    if len(outputs) > 0:
                        bbox_tlwh = []
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        imageSrc = draw_boxes(imageSrc, bbox_xyxy, identities)
                        #for bb_xyxy in bbox_xyxy:
                        #    bbox_tlwh.append(deepsort._xyxy_to_tlwh(bb_xyxy))
                        #results.append((idx, bbox_tlwh, identities))

                    end = time.time()

                    # logging
                    #self.logger.info("time: {:.03f}s, fps_E2E: {:.03f},fps_Track: {:.03f} detection numbers: {}, tracking numbers: {}" \
                    #                .format(end - start, 1 / (end - start),1 / (end - start_tracking), new_bbox.shape[0], len(outputs)))
                                        # save results