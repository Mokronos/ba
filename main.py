import sys
sys.path.insert(1, "./yolo/")
from test_single_imag import yolodet
from helper import *
import argparse

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image", type=str, 
                        help="The path of the input image.")
parser.add_argument("--input_video", type=str, 
                        help="The path of the input video.")
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
input_image = args.input_image
input_video = args.input_video
new_size = args.new_size
letterbox = args.letterbox_resize
class_name_path = args.class_name_path
restore_path = args.restore_path
    
def main(): 
    create_folders(input_video)
    save_frames(input_video)
    for i in range(get_total_frames(input_video)):
        print("--------------------------------" + str(i))
        print(frame_path(input_video,i))
        bbox, conf, cla = yolodet(anchor_path, frame_path(input_video,i), new_size, letterbox, class_name_path, restore_path)
        write_bbox(cla,conf,bbox,i,video_path(extract_name(input_video)) + "/detbbox")


def compare():
    img_all = video_array("./tdata/bahn_1s.mp4")
    vid = cv2.VideoCapture("./tdata/bahn_1s.mp4")
    print(np.array(img_all).shape)
    cv2.imwrite("./berry.png",img_all[0])
    img_berry = cv2.imread("./berry.png")
    print(np.array_equal(img_berry, img_all[0]))
    for i in range(get_total_frames("./tdata/bahn_1s.mp4")):
        _, img_vid = vid.read()
#        img_img = cv2.imread("./data/bahn_1s/data/bahn_1s" + str(i) +".jpg")
        #cv2.imshow("1",img_vid)
        #cv2.imshow("2",img_img)
        cv2.waitKey(0)
        print(i)
        #print(img_vid)
        #print(img_img)
        print(np.array_equal(img_vid,img_all[i]))
#        print(np.array_equal(img_vid,img_img))

#compare()
main()


#bbox, conf, cla = yolodet(anchor_path, "./data/bahn_3s/data/bahn_3s0.jpg", new_size, letterbox, class_name_path, restore_path)
#bbox, conf, cla = yolodet(anchor_path, "./data/bahn_3s/data/bahn_3s1.jpg", new_size, letterbox, class_name_path, restore_path)
#bbox, conf, cla = yolodet(anchor_path, input_image, new_size, letterbox, class_name_path, restore_path)

#print(yolodet(anchor_path, image_path, new_size, letterbox, class_name_path, restore_path))

#print(bbox)
#print(conf)
#print(cla)
#print(get_total_frames("./tdata/test.mp4"))
#create_txt("34 " + arr_str(cla) + arr_str(conf) + arr_str(bbox), "./", "hallo")
#for i in range(3):
 #   write_bbox(cla,conf,bbox, i, "./")
#read_bbox("./bbox.txt")


