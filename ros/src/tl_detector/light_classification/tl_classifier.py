
import rospy
import tensorflow as tf
from styx_msgs.msg import TrafficLight

import numpy as np
import os
import shutil
import cv2
import glob

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.COLOR_LIST = []

        # SSD_GRAPH_FILE = '/home/student/CarND-Capstone/ros/src/tl_detector/Trained_model/frozen_inference_graph.pb'
        SSD_GRAPH_FILE = 'Trained_model/frozen_inference_graph.pb'

        self.detection_graph = self.load_graph(SSD_GRAPH_FILE)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        height,width,_ = image.shape

        # Prediction on cropped image

        y1 = 100
        y2 = 600
        x1 = 100
        x2 = 700
        
        img_crp = image[y1:, x1:x2]
        pr_state = self.prediction(img_crp)

        return pr_state


    def prediction(self, image_cv2):

        # Colors (one for each class)
        cmap = ImageColor.colormap

        self.COLOR_LIST = sorted([c for c in cmap.keys()])

        image = Image.fromarray(image_cv2)
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        
        with tf.Session(graph=self.detection_graph) as sess:
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})
            state = -1
            sign = 'UN'
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.6 
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.

            height, width, _ = np.shape(image)#.size
            box_coords = self.to_image_coords(boxes, height, width)

            img_light = []
            threshold = 220
            i=0

            state = -1
            sign = 'UN'
            
            img = np.copy(image_cv2)
            while (i<len(box_coords)):

                
                ymin = int(box_coords[i][0])
                xmin = int(box_coords[i][1])
                ymax = int(box_coords[i][2])
                xmax = int(box_coords[i][3])

                img_crp = image_cv2[ymin:ymax,xmin:xmax]
                img_crp = img_crp[:,:,::-1]


                # Image thresholding based on Red and Green image channels
                img_crp_mask = np.zeros_like(img_crp[:,:,0])
                img_crp_mask[(img_crp[:,:,0] > threshold) & (img_crp[:,:,1] < threshold)] = 1

                sum_ver = np.sum(img_crp_mask, axis=1)
                sum_len = np.shape(sum_ver)[0]
                tl_portion = sum_len//3

                top = np.sum(sum_ver[:tl_portion])
                mid = np.sum(sum_ver[tl_portion:2*tl_portion])
                bot = np.sum(sum_ver[2*(tl_portion):])


                pixel_thres = 20
                if top > pixel_thres:
                    state = 0
                    # rospy.loginfo("Predicted State RED - %s",scores[i])
                    return state


                # if state == -1:
                    # rospy.loginfo("Predicted State UNKNOWN- %s",scores[i])
 
                i=i+1

                # To Store prediction results

                # img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),thickness=2)

                # time = rospy.Time().now().to_sec()
                # pred_path = '/home/student/CarND-Capstone/imgs/Pred_Imgs/' 
                # fig_name = 'fig_'+sign+'_'+str(time)+'.jpg'
                # image_name = 'Img_'+sign+'_'+str(time)+'.jpg'

                # fig,axs = plt.subplots(2,1)
                # axs[0].imshow(img_crp)
                # axs[1].plot(sum_ver)
                # fig.savefig(pred_path+fig_name)
            
                # plt.close(fig)


                # print(state)
            # if i>0:
            #     # time = rospy.Time().now().to_sec()
            #     cv2.imwrite(pred_path+image_name,img)


            return state #image, classes, scores, 


    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size1
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def draw_boxes(self, image, boxes, classes, thickness=4):
        """Draw bounding boxes on the image"""
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            color = self.COLOR_LIST[class_id]
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                writer =tf.summary.FileWriter('out', graph)
                writer.close()
        return graph