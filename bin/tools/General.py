from PyQt5.QtWidgets import QWidget,QLineEdit,QInputDialog,QGridLayout,QPushButton, QLabel, QFileDialog,QMenu,QAction
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap,QImage
import cv2, os, yaml,time,re
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from bin.tools import RemoteSSH, Draw
import xml.etree.ElementTree as ET


connector = None

work_path, gpu_id, num_gpus, config_file, screen_name = '', None, None, None, None
threshold = 0.5

n_times_iter = 1    
timer = None
pretrained_model = 'model_test.pth'


def General():

    def get_connect():
        global connector
        if connector:
            connector.close()
            connector = None
            btn1.setText('Get')
            btn1.setStyleSheet("background-color: rgb(228, 239, 249);")

        elif le.text():
            host = le.text().split(':')[0]
            port = int(le.text().split(':')[1].split('@')[0])
            #print(le.text())
            if le.text().find('#') != -1:
                user = le.text().split(':')[1].split('@')[1].split('#')[0]
                id_rsa_path = le.text().split('#')[1]
                connector = RemoteSSH.ssh_connect_key(host, port, user, id_rsa_path)
            elif le.text().find('/'):
                user = le.text().split(':')[1].split('@')[1].split('/')[0]
                password = le.text().split(':')[1].split('@')[1].split('/')[1]
                connector = RemoteSSH.ssh_connect(host, port, user, password)
                
            if connector:
                btn1.setText('√')
                btn1.setStyleSheet("background-color: rgb(0, 255, 0);")
            else :
                print('Error ip')
                btn1.setText('Get')
                btn1.setStyleSheet("background-color: rgb(228, 239, 249);")
        else :
            print('Error ip')
            btn1.setText('Get')
            btn1.setStyleSheet("background-color: rgb(228, 239, 249);")
    
    def get_info_yaml():
        # 读取yaml文件
        with open('config/default/info.yaml', 'r') as f:
            info = yaml.load(f, Loader=yaml.FullLoader)
        return info['CONNCECTOR']['data']
    
    '''通用'''
    wgt1 = QWidget()
    layout = QGridLayout()

    line1 = QGridLayout()
    layout.addLayout(line1, 0, 0, alignment=Qt.AlignCenter)

    line2 = QGridLayout()
    layout.addLayout(line2, 1, 0, alignment=Qt.AlignCenter)


    label1 = QLabel('WELCOME TO USE Alice FastLabel',wgt1)
    label1.setFont(QFont("Roman times", 20, QFont.Bold))
    line1.addWidget(label1, 0, 0, alignment=Qt.AlignCenter)

    le = QLineEdit(get_info_yaml() ,wgt1)
    le.setFixedWidth(500)
    line2.addWidget(le, 0, 0, alignment=Qt.AlignCenter)

    btn1 = QPushButton('Get',wgt1)
    btn1.clicked.connect(get_connect)
    line2.addWidget(btn1, 0, 1, alignment=Qt.AlignCenter)

    # self.vedio.setPixmap(img)
    # self.vedio.setAlignment(Qt.AlignCenter)
    # self.vedio.setObjectName('myVedio')
    # self.vedio.setStyleSheet("font: bold; font-size:10px; color: rgb(241, 70, 62); background-color: black")
    wgt1.setLayout(layout)

    return wgt1


work_folder = ''

# name
work_image_list = []
work_image_index = 0
change_image_list = []
new_data_Classes = []

# upper path
work_imageSets_path = ''
work_image_path = ''
work_annotations_path = ''

add_train_mod = 0 # -1 means not add train but use base_train_val, 0 means auto add train, 1 means only click save to add train

def StartLabel():

    def openFile():
        imgName, imgType = QFileDialog.getOpenFileName(wgt1, '打开图片', '', '*.jpg;;*.png;;All Files(*)')
        return imgName, imgType

    def openFolder():
        dir_ = QFileDialog.getExistingDirectory(wgt1, 'Select a folder:', '', QFileDialog.ShowDirsOnly)
        return dir_

    def showImage(imgName):
        
        jpg = QPixmap(imgName).scaled(imageLabel.width(), imageLabel.height())
        imageLabel.setPixmap(jpg)
    
    def showImageFromarray(imgName):
        img = use_voc_format_annotations_draw_image_2(imgName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.imshow('img', img)
        img = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
        jpg = QPixmap.fromImage(img).scaled(imageLabel.width(), imageLabel.height())
        jpg = jpg.scaled(img.size(), Qt.IgnoreAspectRatio)
        # make sure the image in left top
        imageLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        imageLabel.setPixmap(jpg)

    def use_voc_format_annotations_draw_image(imgName):
        img = cv2.imread(work_image_path + imgName)
        xml_path = work_annotations_path + imgName.split('.')[0] + '.xml'
        print(xml_path)
        if not os.path.exists(xml_path):
            return img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(img))
        draw = ImageDraw.Draw(img)
        f = open(xml_path, 'r', encoding='utf-8')
        tree = ET.parse(f)
        root = tree.getroot()
        for object in root.findall('object'):
            name = object.find('name').text
            bndbox = object.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            draw.rectangle((xmin, ymin, xmax, ymax), outline='red')
            draw.text((xmin, ymin), name, fill='red')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def use_voc_format_annotations_draw_image_2(imgName):
        img = cv2.imread(work_image_path + imgName)
        xml_path = work_annotations_path + imgName.split('.')[0] + '.xml'
        print(xml_path)
        if not os.path.exists(xml_path):
            return img
        print('run')
        imageLabel.from_voc_init(xml_path)
        return img
        

    def work():
        imgName, _ = openFile()
        global work_folder
        work_folder = imgName.split('JPEGImages')[0]

        global work_imageSets_path
        work_imageSets_path = work_folder + 'ImageSets/Main/'
        global work_image_path
        work_image_path = work_folder + 'JPEGImages/'
        global work_annotations_path
        work_annotations_path = work_folder + 'Annotations/'
        
        global work_image_list
        work_image_list = os.listdir(work_image_path)

        now_image_name = imgName.split('/')[-1]

        global work_image_index
        work_image_index = work_image_list.index(now_image_name)

        imageLabel.re_init()
        #print(work_folder)
        showImageFromarray(now_image_name)

    def select_another_xml_folder():
        global work_annotations_path
        work_annotations_path = openFolder() + '/'

    def get_pre_image():
        global work_image_index
        global work_image_list
        global add_train_mod
        if work_image_index == 0:
            print('This is the first image')
        else:
            save_xml()
            if add_train_mod == 0:
                add_train_img(work_image_index)
            imageLabel.re_init()
            work_image_index -= 1
            showImageFromarray(work_image_list[work_image_index])
    
    def get_next_image():
        global work_image_index
        global work_image_list
        global add_train_mod

        if work_image_index == len(work_image_list)-1:
            print('This is the last image')
        else:
            save_xml()
            if add_train_mod == 0:
                add_train_img(work_image_index)
            imageLabel.re_init()
            work_image_index += 1
            showImageFromarray(work_image_list[work_image_index])

    def add_train_img(add_index):
        if add_index not in change_image_list:
            change_image_list.append(add_index)

    def save_xml():
        global work_image_index
        global work_image_list
        global work_image_path
        global change_image_list
        imageLabel.convert_voc(work_image_path+work_image_list[work_image_index], work_annotations_path+work_image_list[work_image_index].split('.')[0]+'.xml')

    def click_save_xml():
        global work_image_index
        global work_image_list
        global work_image_path
        global change_image_list
        global add_train_mod
        if add_train_mod == 0 or add_train_mod == 1:
            add_train_img(work_image_index)
        imageLabel.convert_voc(work_image_path+work_image_list[work_image_index], work_annotations_path+work_image_list[work_image_index].split('.')[0]+'.xml')
    
    def myfilter():
        wgt_temp = QWidget()
        text, ok = QInputDialog.getText(wgt_temp, 'info', '请输入过滤器：', QLineEdit.Normal, imageLabel.get_filter())
        if ok and type(eval(text)) == list:
            imageLabel.set_filter(eval(text))
        else:
            imageLabel.set_filter([])
        global new_data_Classes
        new_data_Classes = eval(imageLabel.get_filter())

    def clear_global_param():
        global work_folder, work_image_list, work_image_index, change_image_list, new_data_Classes, work_imageSets_path, work_image_path, work_annotations_path
        work_folder, work_image_list, work_image_index, change_image_list, new_data_Classes, work_imageSets_path, work_image_path, work_annotations_path = '', [], 0, [], [], '', '', ''
        global work_path, gpu_id, num_gpus, config_file, screen_name, n_times_iter
        work_path, gpu_id, num_gpus, config_file, screen_name = '', None, None, None, None
        n_times_iter = 1

    def contextMenuEvent(event):
        global add_train_mod
        cmenu = QMenu(wgt1)
        Use_Self = cmenu.addAction('Use Self')
        Auto = cmenu.addAction('Auto Save')
        Click = cmenu.addAction('Only Click')
        yPos = event.pos().y()
        print(type(yPos))
        if yPos > 734:        
            action = cmenu.exec_(wgt1.mapToGlobal(event.pos()))
            if action == Use_Self:
                btn5.setText('Use Self')
                add_train_mod = -1
            elif action == Auto:
                btn5.setText('Auto Save')
                add_train_mod = 0
            elif action == Click:
                btn5.setText('Only Click')
                add_train_mod = 1
        else :
            print('no action')

    def keyPressEvent(event):
        global add_train_mod
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_D:
            imageLabel.del_rect()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_U:
            btn5.setText('Use Self')
            add_train_mod = -1
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_A:
            btn5.setText('Auto Save')
            add_train_mod = 0
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_O:
            btn5.setText('Only Click')
            add_train_mod = 1
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            click_save_xml()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_J:
            work()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_X:
            select_another_xml_folder()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_F:
            myfilter()
        
        if event.key() == Qt.Key_Left:
            get_pre_image()
        elif event.key() == Qt.Key_Right:
            get_next_image()
            
    wgt1 = QWidget()
    wgt1.setFocusPolicy(Qt.StrongFocus)
    wgt1.contextMenuEvent = contextMenuEvent
    wgt1.keyPressEvent = keyPressEvent

    layout = QGridLayout()

    line1 = QGridLayout()
    layout.addLayout(line1, 0, 0, alignment=Qt.AlignCenter)

    line2 = QGridLayout()
    layout.addLayout(line2, 1, 0, alignment=Qt.AlignCenter)

    #imageLabel = QLabel(wgt1)
    imageLabel = Draw.myDrawRect(QLabel(wgt1))
    imageLabel.setFixedSize(1280, 720)

    imageLabel.setStyleSheet("border: 1px solid black")
    line1.addWidget(imageLabel, 0, 0, alignment=Qt.AlignCenter)

    btn1 = QPushButton('Select',wgt1)
    btn1.clicked.connect(work)
    line2.addWidget(btn1, 0, 0, alignment=Qt.AlignCenter)

    # 预留接口，用于直接生成label 和 bbox
    btn2 = QPushButton('Select xml_folder',wgt1)
    btn2.clicked.connect(select_another_xml_folder)
    line2.addWidget(btn2, 0, 1, alignment=Qt.AlignCenter)

    btn3 = QPushButton('filter',wgt1)
    btn3.clicked.connect(myfilter)
    line2.addWidget(btn3, 0, 2, alignment=Qt.AlignCenter)

    btn3 = QPushButton('<-',wgt1)
    btn3.clicked.connect(get_pre_image)
    line2.addWidget(btn3, 0, 3, alignment=Qt.AlignCenter)

    btn4 = QPushButton('->',wgt1)
    btn4.clicked.connect(get_next_image)
    line2.addWidget(btn4, 0, 4, alignment=Qt.AlignCenter)

    btn5 = QPushButton('Auto Save',wgt1)
    btn5.clicked.connect(click_save_xml)
    line2.addWidget(btn5, 0, 5, alignment=Qt.AlignCenter)

    btn6 = QPushButton('Clear run info',wgt1)
    btn6.clicked.connect(clear_global_param)
    btn6.setStyleSheet("background-color: red")
    line2.addWidget(btn6, 0, 6, alignment=Qt.AlignCenter)


    wgt1.setMouseTracking(True)
    wgt1.setLayout(layout)

    return wgt1    

info_dict = {}

def Statistics():

    def format_read_xml(xml_path):
        ans = []
        with open(xml_path, 'r', encoding='utf-8') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for object in root.findall('object'):
                name = object.find('name').text
                bndbox = object.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                ans.append([name, xmin, ymin, xmax, ymax])
            return ans

    def get_info_from_work_annotations_path():
        global work_annotations_path
        global change_image_list
        global work_image_list
        global info_dict
        info_dict = {'#ClassList': [], 'total_image_nums':0, '#Image_nums': 0}

        info_dict['total_image_nums'] = len(work_image_list)

        # 通过imageName读取所有的xml文件
        for index in change_image_list:
            xml_path = work_annotations_path+work_image_list[index].split('.')[0]+'.xml'
            # 统计xml文件中的类别
            for prop in format_read_xml(xml_path):
                if prop[0] in info_dict:
                    info_dict[prop[0]] += 1
                else:
                    info_dict[prop[0]] = 1
                    info_dict['#ClassList'].append(prop[0])
            # 统计图片张数
            if '#Image_nums' in info_dict:
                info_dict['#Image_nums'] += 1
            else:
                info_dict['#Image_nums'] = 1

    def show_info():
        get_info_from_work_annotations_path()
        # make info_dict to str, one value one line
        my_txt = ''
        for key in info_dict:
            my_txt += key + ': ' + str(info_dict[key]) + '\n'
    
        imageLabel.setText(my_txt)
            
    def generate_train_val():
        global work_imageSets_path
        global new_data_Classes
        global change_image_list
        global work_image_list
        global work_annotations_path
        global add_train_mod


        if not os.path.exists(work_imageSets_path):
            os.makedirs(work_imageSets_path)

        if add_train_mod == -1:
            with open(work_imageSets_path+'ClassNames.txt', 'r') as f:
                if new_data_Classes != []:
                    info_dict['#ClassList'] = new_data_Classes
                else:
                    info_dict['#ClassList'] = f.readlines()

            with open(work_imageSets_path+'train.txt', 'r') as f:
                work_image_list = f.readlines()
        
        else:
            with open(work_imageSets_path+'ClassNames.txt', 'w') as f:
                if new_data_Classes == []:
                    for i in info_dict['#ClassList']:
                        f.write(i + '\n')
                else:
                    # print(type(new_data_Classes))
                    # print(type(info_dict['#ClassList']))
                    # print(new_data_Classes)
                    # print(info_dict['#ClassList'])
                    
                    for i in new_data_Classes:
                        if i in info_dict['#ClassList']:
                            f.write(i + '\n')

            with open(work_imageSets_path+'train.txt', 'w') as f:
                for i in change_image_list:
                    f.write(work_image_list[i].split('.')[0] + '\n')

            with open(work_imageSets_path+'val.txt', 'w') as f:
                for i in range(len(work_image_list)):
                    if i not in change_image_list:
                        f.write(work_image_list[i].split('.')[0] + '\n')

    def read_yaml(path='config/my_config.yaml'):
        with open(path, 'r',encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        
        global work_path
        global gpu_id
        global num_gpus
        global config_file
        global screen_name
        global n_times_iter
        global threshold

        work_path = data['work_path']
        gpu_id = data['gpu_id']
        num_gpus = len(eval(gpu_id))
        config_file = data['config_file']
        screen_name = data['screen_name']
        n_times_iter = data['n_times_iter']
        threshold = data['threshold']

    def update_remote_data_prepare():
        read_yaml()

        global work_annotations_path
        global work_imageSets_path
        global work_path
        RemoteSSH.update_annotations_and_ImageSets_Main(connector, work_annotations_path, work_imageSets_path, work_path)

    def Clear_remote_output_inference():
        global work_path
        RemoteSSH.clear_output_inference_files(connector, work_path)

    def Quick_click():
        show_info()
        generate_train_val()
        update_remote_data_prepare()
        Clear_remote_output_inference()

    wgt1 = QWidget()
    layout = QGridLayout()

    line1 = QGridLayout()
    layout.addLayout(line1, 0, 0, alignment=Qt.AlignCenter)

    line2 = QGridLayout()
    layout.addLayout(line2, 1, 0, alignment=Qt.AlignCenter)


    imageLabel = QLabel(wgt1)
    imageLabel.setFixedSize(1280, 720)
    imageLabel.setStyleSheet("border: 1px solid black")
    line1.addWidget(imageLabel, 0, 0, alignment=Qt.AlignCenter)

    btn1 = QPushButton('Get info',wgt1)
    btn1.clicked.connect(show_info)
    line2.addWidget(btn1, 0, 0, alignment=Qt.AlignCenter)

    btn2 = QPushButton('Generate train/val',wgt1)
    btn2.clicked.connect(generate_train_val)
    line2.addWidget(btn2, 0, 1, alignment=Qt.AlignCenter)

    btn3 = QPushButton('Update Remote data prepare',wgt1)
    btn3.clicked.connect(update_remote_data_prepare)
    line2.addWidget(btn3, 0, 2, alignment=Qt.AlignCenter)

    btn4 = QPushButton('Clear inference',wgt1)
    btn4.clicked.connect(Clear_remote_output_inference)
    line2.addWidget(btn4, 0, 3, alignment=Qt.AlignCenter)

    btn5 = QPushButton('Quick click',wgt1)
    btn5.clicked.connect(Quick_click)
    btn5.setStyleSheet("background-color: rgb(0, 255, 0);")
    line2.addWidget(btn5, 0, 4, alignment=Qt.AlignCenter)

    wgt1.setLayout(layout)

    return wgt1   

def Coco2Voc():
    wgt1 = QWidget()
    return wgt1

def Inference():

    def read_yaml(path='config/my_config.yaml'):
        with open(path, 'r',encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        
        global work_path
        global gpu_id
        global num_gpus
        global config_file
        global screen_name
        global n_times_iter
        global threshold
        global pretrained_model

        work_path = data['work_path']
        gpu_id = data['gpu_id']
        num_gpus = len(eval(gpu_id))
        config_file = data['config_file']
        screen_name = data['screen_name']
        n_times_iter = float(data['n_times_iter'])
        threshold = data['threshold']
        pretrained_model = data['pretrained_model']

    def run_ft():

        read_yaml()

        global connector
        global work_path
        global gpu_id
        global num_gpus
        global config_file
        global screen_name
        global change_image_list
        global pretrained_model

        num_classes = 0
        if new_data_Classes == []:
            num_classes = len(info_dict['#ClassList'])
        else:
            num_classes = len(new_data_Classes)
        
        RemoteSSH.clear_output_inference_files(connector, work_path, 'log.txt')

        #print(work_path, gpu_id, num_gpus, config_file, screen_name)
        RemoteSSH.make_yaml(int(len(change_image_list)*n_times_iter), num_classes, 'config/'+config_file) # change
        RemoteSSH.make_run_sh(connector, work_path, gpu_id, num_gpus, config_file, screen_name, pretrained_model)
        global timer
        timer = QTimer(wgt1)
        timer.timeout.connect(get_log_info_then_analysis)
        timer.stop()
        timer.start(500)

    def run_inference():
        read_yaml()
        
        global connector
        global work_path
        global gpu_id
        global num_gpus
        global config_file
        global screen_name
        global change_image_list
        global threshold

        RemoteSSH.clear_output_inference_files(connector, work_path)

        print(work_path, gpu_id, num_gpus, config_file, screen_name)
        #RemoteSSH.make_yaml(len(change_image_list))
        RemoteSSH.make_inference_sh(connector, work_path, gpu_id, config_file, screen_name, threshold) # change
        global timer
        timer = QTimer(wgt1)
        timer.timeout.connect(get_log_info_then_analysis_inference)
        timer.stop()
        timer.start(500)

    def get_log_info_then_analysis():
        global connector
        global work_path
        global timer
        info = RemoteSSH.get_log_info(connector,work_path)
        label2.setText(info)
        if info.find('Total training time') != -1:
            if timer != None:
                timer.stop()
                timer = None


    def get_log_info_then_analysis_inference():
        global connector
        global work_path
        global timer
        info = RemoteSSH.get_log_info(connector,work_path,'/output/inference/inference_log.txt', tail_num=10)
        label4.setText(info)
        if info.find('Inference finished on') != -1:
            if timer != None:
                timer.stop()
                timer = None

    def get_own_inferece_xml():
        global connector
        global work_path
        global work_folder
        RemoteSSH.get_inference_xml(connector,work_path,work_folder)


    def clear_screen():
        global connector
        global screen_name
        read_yaml()
        out = RemoteSSH.find_screen(connector, screen_name)
        if out != '':
            screen_id_list = out.split('\n')
            for screen_id_line in screen_id_list:
                if screen_id_line == '':
                    continue
                screen_id_str = screen_id_line.split('.')[0]
                screen_id = re.findall(r'\d+', screen_id_str)[0]
                RemoteSSH.kill_screen(connector, screen_id)


    wgt1 = QWidget()
    layout = QGridLayout()

    line1 = QGridLayout()
    layout.addLayout(line1, 0, 0, alignment=Qt.AlignLeft)

    line2 = QGridLayout()
    layout.addLayout(line2, 1, 0, alignment=Qt.AlignCenter)

    line3 = QGridLayout()
    layout.addLayout(line3, 2, 0, alignment=Qt.AlignLeft)

    line4 = QGridLayout()
    layout.addLayout(line4, 3, 0, alignment=Qt.AlignCenter)

    line5 = QGridLayout()
    layout.addLayout(line5, 4, 0, alignment=Qt.AlignCenter)

    label1 = QLabel('Fine-True-info:',wgt1)
    line1.addWidget(label1, 0, 0, alignment=Qt.AlignLeft)

    btn1_1 = QPushButton('Run:',wgt1)
    btn1_1.clicked.connect(run_ft)
    line1.addWidget(btn1_1, 0, 2, alignment=Qt.AlignRight)

    btn1_2 = QPushButton('Clear run:', wgt1)
    btn1_2.clicked.connect(clear_screen)
    line1.addWidget(btn1_2, 0, 3, alignment=Qt.AlignRight)

    label2 = QLabel('info:', wgt1)
    label2.setFixedSize(1280, 300)
    label2.setStyleSheet("border: 1px solid black")
    line2.addWidget(label2, 0, 0, alignment=Qt.AlignCenter)

    label3 = QLabel('Inference:',wgt1)
    line3.addWidget(label3, 0, 0, alignment=Qt.AlignLeft)

    btn3_1 = QPushButton('Run:',wgt1)
    btn3_1.clicked.connect(run_inference)
    line3.addWidget(btn3_1, 0, 2, alignment=Qt.AlignCenter)

    btn3_2 = QPushButton('Clear run:', wgt1)
    btn3_2.clicked.connect(clear_screen)
    line3.addWidget(btn3_2, 0, 3, alignment=Qt.AlignCenter)

    btn3_3 = QPushButton('GetOwnXml:', wgt1)
    btn3_3.clicked.connect(get_own_inferece_xml)
    line3.addWidget(btn3_3, 0, 4, alignment=Qt.AlignCenter)

    label4 = QLabel('info:', wgt1)
    label4.setFixedSize(1280, 300)
    label4.setStyleSheet("border: 1px solid black")
    line4.addWidget(label4, 0, 0, alignment=Qt.AlignCenter)

    wgt1.setLayout(layout)

    return wgt1   

