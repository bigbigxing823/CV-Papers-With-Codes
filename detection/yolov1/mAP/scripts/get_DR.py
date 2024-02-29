import os.path
import cv2

def xywh2xyxy(x, y, w, h, height, width):
    x1 = max(int(x - w/2), 0)
    y1 = max(int(y - h/2), 0)
    x2 = min(int(x + w/2), width)
    y2 = min(int(y + h/2), height)
    return x1, y1, x2, y2


cls_txt_path = r'C:\Users\Administrator\Desktop\my_paper\code\yolov1\data\NEU-DET\classes.txt'
class_txt = open(cls_txt_path, 'r')
class_dict = {}
for id_, line in enumerate(class_txt.readlines()):
    line = line.strip()
    class_dict[id_] = line

result_dir = r'C:\Users\Administrator\Desktop\my_paper\code\yolov1\result'
dst_dir = r'C:\Users\Administrator\Desktop\my_paper\code\yolov1\mAP\input\detection-results'
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

image_dir = r'C:\Users\Administrator\Desktop\my_paper\code\yolov1\data\NEU-DET\JPEGImages'
result_list = os.listdir(result_dir)
for result_name in result_list:
    result_path = result_dir + '/' + result_name
    result_txt = open(result_path, 'r')
    dst_txt = open(os.path.join(dst_dir, result_name), 'w')
    image_path = os.path.join(image_dir, result_name.split('.')[0]+'.jpg')
    image_data = cv2.imread(image_path)
    height, width, _ = image_data.shape
    print(result_name)
    for line in result_txt.readlines():
        line = line.strip()
        Cls, x1, y1, x2, y2, prob = line.split(' ')
        x1, y1, x2, y2 = int(float(x1) * width), int(float(y1) * height), int(float(x2) * width), int(float(y2) * height)
        if int(Cls) < 6:
            dst_txt.writelines(' '.join([class_dict[int(Cls)], str(prob), str(x1), str(y1), str(x2), str(y2)]) + '\n')
        # cv2.rectangle(image_data, (x1, y1), (x2, y2), (0, 0, 255))
        # cv2.imshow('ret', image_data)
        # cv2.waitKey()

