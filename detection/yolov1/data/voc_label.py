# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(data_path, image_id):
    in_file = open(data_path+'/Annotations/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open(data_path+'/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    if root.iter('object'):
        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        return True
    else:
        return False


if __name__ == '__main__':
    cur_path = os.getcwd()
    sets = ['train', 'val', 'test']
    classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    data_path = cur_path + r'/NEU-DET'
    for image_set in sets:
        if not os.path.exists(os.path.join(data_path, 'labels')):
            os.makedirs(os.path.join(data_path, 'labels'))
        image_ids = open(data_path + '/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
        list_file = open(data_path+'/%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            flag = convert_annotation(data_path, image_id)
            if flag:
                list_file.write(f'{data_path}/JPEGImages/{image_id}.jpg {data_path}/labels/{image_id}.txt\n')
        list_file.close()
