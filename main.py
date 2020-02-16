import sys
sys.path.insert(1, "./yolo/")
from test_single_imag import yolodet
from helper import *
import argparse

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("input_image", type=str,
                        help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./yolo/data/yolo_anchors.txt",
                        help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                        help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./yolo/data/coco.names",
                        help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./yolo/data/best_model",
                        help="The path of the weights to restore.")
args = parser.parse_args()
    
    
anchor_path = args.anchor_path
image_path = args.input_image
new_size = args.new_size
letterbox = args.letterbox_resize
class_name_path = args.class_name_path
restore_path = args.restore_path
    
    
    
    
#bbox, conf, cla = yolodet(anchor_path, image_path, new_size, letterbox, class_name_path, restore_path)
#print(yolodet(anchor_path, image_path, new_size, letterbox, class_name_path, restore_path))

#print(bbox)
#print(conf)
#print(cla)
print(get_total_frames("./tdata/test.mp4"))


