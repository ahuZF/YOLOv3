import xml.etree.ElementTree as ET
from os import getcwd

#sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes=["aircraft"]

'''def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)  ##分析指定xml文件
    root = tree.getroot()   ###获取根节点

    for obj in root.iter('object'):    ### 获取所以根节点里面的‘object'
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        print(image_id)
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()'''

def convert_annotation(image_id, list_file):
    in_file = open(r'G:\Deep\yolo\yolo3-keras-master\yolo3-keras-master\voc_rs\aircraft\aircraft\Annotation\xml\%s.xml'%(image_id))
    #print(in_file)
    tree=ET.parse(in_file)  ##分析指定xml文件
    root = tree.getroot()   ###获取根节点

    for obj in root.iter('object'):    ### 获取所以根节点里面的‘object'
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        print(image_id)
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

image_ids = open(r'G:\Deep\yolo\yolo3-keras-master\yolo3-keras-master\voc_rs\aircraft\aircraft\IMG_SETS\train.txt').read().strip().split()
list_file = open('air.txt', 'w')
for image_id in image_ids:
    list_file.write(r'G:\Deep\yolo\yolo3-keras-master\yolo3-keras-master\voc_rs\aircraft\aircraft\JPEGImages\%s.jpg '%(image_id))
    #print(image_id)
    convert_annotation(image_id, list_file)
    list_file.write('\n')
list_file.close()

