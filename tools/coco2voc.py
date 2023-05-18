# -*- coding: utf-8 -*-
from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

savepath = "VOC2012_coco_convert_full_80/" # 此处为存储路径
# create the folder savepath
if not os.path.exists(savepath):
    os.mkdir(savepath)
datasets_list = ['val2017']  # 这里设置你要转换的数据集，我这里只转换了val2017
# 注意 这里默认你的coco数据集的目录结构为：coco/annotations/instances_val2017.json
#                                       coco/annotations/instances_train2017.json
#                                       coco/train2017/*.jpg
#                                       coco/val2017/*.jpg
# 因此如为自定义数据集，请遵循此目录结构，json文件的命名方法为instances_+数据集名称+.json
img_dir = savepath + 'JPEGImages_me/'  # 这里时处理后图片存放位置
anno_dir = savepath + 'Annotations_me/'  # 这里是处理后标签存放位置
# 下面的classes_names是你要转换的类别列表
classes_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#classes_names = ['airplane','bicycle', 'bird','boat', 'bottle','bus',   'car', 'cat', 'chair','cow','dining table','dog','horse','motorcycle', 'person','potted plant', 'sheep', 'couch','train', 'tv']
#classes_names = ['cat', 'tv']

# classes_names = ['truck',
#  'traffic light',
#  'fire hydrant',
#  'stop sign',
#  'parking meter',
#  'bench',
#  'elephant',
#  'bear',
#  'zebra',
#  'giraffe',
#  'backpack',
#  'umbrella',
#  'handbag',
#  'tie',
#  'suitcase',
#  'frisbee',
#  'skis',
#  'snowboard',
#  'sports ball',
#  'kite',
#  'baseball bat',
#  'baseball glove',
#  'skateboard',
#  'surfboard',
#  'tennis racket',
#  'wine glass',
#  'cup',
#  'fork',
#  'knife',
#  'spoon',
#  'bowl',
#  'banana',
#  'apple',
#  'sandwich',
#  'orange',
#  'broccoli',
#  'carrot',
#  'hot dog',
#  'pizza',
#  'donut',
#  'cake',
#  'bed',
#  'toilet',
#  'laptop',
#  'mouse',
#  'remote',
#  'keyboard',
#  'cell phone',
#  'microwave',
#  'oven',
#  'toaster',
#  'sink',
#  'refrigerator',
#  'book',
#  'clock',
#  'vase',
#  'scissors',
#  'teddy bear',
#  'hair drier',
#  'toothbrush']

dataDir = 'coco'  ####### 连接到coco的数据集
headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''


def mkr(path):
    if os.path.exists(path):
        #shutil.rmtree(path)
        #os.mkdir(path)
        pass
    else:
        os.mkdir(path)


mkr(img_dir)
mkr(anno_dir)


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def write_xml(anno_path, head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr % (obj[0], obj[1], obj[2], obj[3], obj[4]))
    f.write(tail)


def save_annotations_and_imgs(coco, dataset, filename, objs):
    anno_path = anno_dir + filename[:-3] + 'xml'
    print('anno_path:%s' % anno_path)
    # img_path=dataDir+'/'+'images'+'/'+dataset+'/'+filename
    if dataset != 'val2017':
        dataset_temp = 'train2017'
    else:
        dataset_temp = 'val2017'
    img_path = dataDir + '/' + dataset_temp + '/' + filename
    print('img_path:%s' % img_path)
    print('step3-image-path-OK')
    dst_imgpath = img_dir + filename

    img = cv2.imread(img_path)
    '''if (img.shape[2] == 1):
        print(filename + " not a RGB image")     
        return'''
    print('img_path:%s' % img_path)
    print('dst_imgpath:%s' % dst_imgpath)
    #shutil.copy(img_path, dst_imgpath)

    head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path, head, objs, tail)


def showimg(coco, dataset, img, classes, cls_id, show=True):
    global dataDir
    if dataset != 'val2017':
        dataset_temp = 'train2017'
    else :
        dataset_temp = 'val2017'
    # I=Image.open('%s/%s/%s/%s'%(dataDir,'images',dataset,img['file_name']))
    I = Image.open('%s/%s/%s' % (dataDir, dataset_temp, img['file_name']))  ########may be you can changed
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        class_name = classes[ann['category_id']]
        if class_name in classes_names:
            print(class_name)
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                # draw = ImageDraw.Draw(I)
                # draw.rectangle([xmin, ymin, xmax, ymax])
    # if show:
    # plt.figure()
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()
    return objs


for dataset in datasets_list:
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)  # 存放json文件的路径
    print('annFile:%s' % annFile)
    coco = COCO(annFile)
    '''
    loading annotations into memory...
    Done (t=0.81s)
    creating index...
    index created!
    '''
    classes = id2name(coco)
    print("classes:%s" % classes)
    classes_ids = coco.getCatIds(catNms=classes_names)
    print(classes_ids)
    filename_list = []
    for cls in classes_names:
        cls_id = coco.getCatIds(catNms=[cls])
        img_ids = coco.getImgIds(catIds=cls_id)
        print(cls, len(img_ids))
        upperme = min(len(img_ids), 400)
        img_ids_temp = img_ids[0:upperme]
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids_temp):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            if filename not in filename_list:
                my_classes_names = img['file_name']
                #print(filename)
                filename_list.append(filename)
                # write filename to txt
                if dataset == 'val2017':
                    txt_save_path = savepath+'/ImageSets/Main/'
                    txt_file = 'val.txt'

                else:
                    txt_save_path = savepath+'/ImageSets/'+dataset +'/'
                    txt_file = 'train.txt'
                if not os.path.exists(txt_save_path):
                    # create forlder
                    os.makedirs(txt_save_path)
                with open(txt_save_path+txt_file, 'a') as f:
                    f.write(filename.split('.')[0] + '\n')


            # print(filename)
            objs = showimg(coco, dataset, img, classes, classes_ids, show=False)
            # print(objs)
            save_annotations_and_imgs(coco, dataset, filename, objs)
    print(filename_list)
    print(len(filename_list))
