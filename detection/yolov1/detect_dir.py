import os
from detect import DetectModel

if __name__ == '__main__':
    ckpt_path = 'checkpoints/yolov1.pt'
    detector = DetectModel(ckpt_path)

    test_txt = r'C:\Users\Administrator\Desktop\my_paper\code\yolov1\data\NEU-DET\val.txt'
    save_dir = 'result'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_txt = True
    detector.detect_dir(test_txt, False, save_dir, save_txt)