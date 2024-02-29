import os
import cv2


def xywh2xyxy(x, y, w, h):
    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)
    return x1, y1, x2, y2


cls_txt_path = r'C:\Users\Administrator\Desktop\my_paper\code\yolov1\data\NEU-DET\classes.txt'
class_txt = open(cls_txt_path, 'r')
class_dict = {}
for id_, line in enumerate(class_txt.readlines()):
    line = line.strip()
    class_dict[id_] = line
print(class_dict)

gt_txt_path = r'C:\Users\Administrator\Desktop\my_paper\code\yolov1\data\NEU-DET\val.txt'
dst_dir = r'C:\Users\Administrator\Desktop\my_paper\code\yolov1\mAP\input\ground-truth'
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

gt_txt = open(gt_txt_path, 'r')
for line in gt_txt.readlines():
    line = line.strip()
    img_path, lab_path = line.split(' ')[0], line.split(' ')[-1]
    lab_name = os.path.basename(lab_path)
    img_data = cv2.imread(img_path)
    height, width, _ = img_data.shape
    lab_txt = open(lab_path)
    dst_path = os.path.join(dst_dir, lab_name)
    dst_txt = open(dst_path, 'w')
    for line in lab_txt.readlines():
        line = line.strip()
        Cls, x, y, w, h = line.split(' ')
        x, y, w, h = int(float(x)*width), int(float(y)*height), int(float(w)*width), int(float(h)*height)
        x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)
        dst_txt.write(' '.join([class_dict[int(Cls)], str(x1), str(y1), str(x2), str(y2)]) + '\n')


