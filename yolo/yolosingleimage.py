# coding: utf-8
# refer to wizyoungs yolo implementation for how this works
# here i just created a function which takes an imagepath and returns the boundingboxes for that image
# for the yolo parameters i used mainly the default values of below function which can be found in given folder

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

#define root directory of whole project to set absolute paths for files(im calling this function from a subdirectory so relative paths are weird/wrong)
rootpath =r"C:\Users\Sebastian\vim_files\ba" 

# takes paths and letterbox(true/false) (and new_size) --> returns bbox + confidence + class
def yolodet( image_path,anchor_path =rootpath + "/yolo/data/yolo_anchors.txt", new_size = [416, 416], letterbox = True , class_name_path =rootpath + "/yolo/data/coco.names", restore_path = rootpath + "/yolo/data/best_model"):

    
    anchors = parse_anchors(anchor_path)
    classes = read_class_names(class_name_path)
    num_class = len(classes)
    color_table = get_color_table(num_class)
    
    img_ori = cv2.imread(image_path)
    if letterbox:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, new_size[0], new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
        yolo_model = yolov3(num_class, anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    
        pred_scores = pred_confs * pred_probs
    
        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)
    
        saver = tf.train.Saver()
        saver.restore(sess, restore_path)
    
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
    
        # rescale the coordinates to the original image
        if letterbox:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori/float(new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori/float(new_size[1]))
    
    tf.reset_default_graph()
    

    #transform detections into 1 line (#1class,#1conf,#1xmin,#1ymin,#1max,#1ymax,#2class,#2conf,...)
    boxes = []
    for i in range(np.shape(boxes_)[0]):
        boxes.append(labels_[i])
        boxes.append(scores_[i])
        boxes.extend(boxes_[i,:])

    return boxes

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
    parser.add_argument("input_image", type=str,
                        help="The path of the input image.")
    parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                        help="The path of the anchor txt file.")
    parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                        help="Resize the input image with `new_size`, size format: [width, height]")
    parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to use the letterbox resize.")
    parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                        help="The path of the class names.")
    parser.add_argument("--restore_path", type=str, default="./data/best_model",
                        help="The path of the weights to restore.")
    args = parser.parse_args()
    
    
    anchor_path = args.anchor_path
    image_path = args.input_image
    new_size = args.new_size
    letterbox = args.letterbox_resize
    class_name_path = args.class_name_path
    restore_path = args.restore_path
    
    
    
    
    print(yolodet(anchor_path, image_path, new_size, letterbox, class_name_path, restore_path))
