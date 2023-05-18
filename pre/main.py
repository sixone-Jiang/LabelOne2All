from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QVBoxLayout, QWidget,QMessageBox,QGridLayout,QFileDialog,QStyleFactory
from PyQt5.QtGui import QFont, QPixmap,QImage,QIcon
from PyQt5.QtCore import Qt, QTimer
import sys,cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def qtpixmap_to_cvimg(qtpixmap):

    qimg = qtpixmap.toImage()
    temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    temp_shape += (4,)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    result = result[..., :3]

    return result

class myWindow:
    def __init__(self):
        self.initUI()
        self.loc_vedio_click = -1
        self.sl_vedio_click = -1
        self.infer_vedio_click = -1
        self.infer_flag = False
        self.previous_poses = []
        self.track = 1

        self.fps = 30
        self.fps_ratio = 2
        #self.startCapture()

    def initUI(self):
        self.win = QMainWindow()
        self.win.resize(1600, 1400)
        self.win.setGeometry(800, 200, 800, 600)
        self.win.setObjectName("MainWindow")
        self.win.setStyleSheet("#MainWindow{border-image:url(./images/background/1.jpg);}")
        self.win.setWindowTitle("姿态识别")
        self.win.setWindowIcon(QIcon("./images/logo/logo.jpg"))
        self.win.setStyle(QStyleFactory.create('Fusion'))
        #self.win.setWindowFlags(Qt.CustomizeWindowHint) 

        wgt = QWidget()
        layout = QGridLayout()

        self.vedio = QLabel()
        self.vedio.setFixedSize(1280, 720)
        #self.img_list = ['./images/hide.png', './images/next.png','./images/show.png']
        self.img_list = ['./images/background/1.jpg', './pictures/10.jpg','./pictures/1.jpg']
        self.img_pix = [QPixmap(img).scaled(self.vedio.size(), Qt.KeepAspectRatioByExpanding) for img in self.img_list]
        '''
        self.label = QLabel()
        self.label.setFixedSize(700, 500)
        '''
        self.img_id = 0
        img = self.img_pix[self.img_id]
        self.wait_infer_image = img
        '''
        self.label.setPixmap(img)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setObjectName('mylabel')
        self.label.setStyleSheet("font: bold; font-size:10px; color: rgb(241, 70, 62); background-color: green")
        layout.addWidget(self.label, 0, 0, alignment=Qt.AlignCenter)
        '''

        self.vedio.setPixmap(img)
        self.vedio.setAlignment(Qt.AlignCenter)
        self.vedio.setObjectName('myVedio')
        self.vedio.setStyleSheet("font: bold; font-size:10px; color: rgb(241, 70, 62); background-color: black")
        layout.addWidget(self.vedio, 0, 0, alignment=Qt.AlignCenter)

        line2 = QGridLayout()
        layout.addLayout(line2, 1, 0, alignment=Qt.AlignCenter)


        self.btn_loc_next_pic = QPushButton("LocNext pic")
        self.btn_loc_next_pic.setFont(QFont("Ubuntu Mono", 7))
        self.btn_loc_next_pic.setFixedSize(200, 50)
        self.btn_loc_next_pic.clicked.connect(self.btn_next_pic_clicked)
        line2.addWidget(self.btn_loc_next_pic, 0, 0, alignment=Qt.AlignCenter)

        self.btn_slpic = QPushButton("Select pic")
        self.btn_slpic.setFont(QFont("Ubuntu Mono", 7))
        self.btn_slpic.setFixedSize(200, 50)
        self.btn_slpic.clicked.connect(self.btn_slpic_clicked)
        line2.addWidget(self.btn_slpic, 0, 1, alignment=Qt.AlignCenter)
        
        self.btn_infer_pic = QPushButton("Pic Inference")
        self.btn_infer_pic.setFont(QFont("Ubuntu Mono", 7))
        self.btn_infer_pic.setFixedSize(200, 50)
        self.btn_infer_pic.clicked.connect(self.btn_infer_pic_clicked)
        line2.addWidget(self.btn_infer_pic, 0, 2, alignment=Qt.AlignCenter)
        
        self.btn_local_vedio = QPushButton("Local Vedio")
        self.btn_local_vedio.setFont(QFont("Ubuntu Mono", 7))
        self.btn_local_vedio.setFixedSize(200, 50)
        self.btn_local_vedio.clicked.connect(self.btn_loc_vedio_clicked)
        line2.addWidget(self.btn_local_vedio, 0, 3, alignment=Qt.AlignCenter)

        self.btn_sl_vedio = QPushButton("Select Vedio")
        self.btn_sl_vedio.setFont(QFont("Ubuntu Mono", 7))
        self.btn_sl_vedio.setFixedSize(200, 50)
        self.btn_sl_vedio.clicked.connect(self.btn_slvedio_clicked)
        line2.addWidget(self.btn_sl_vedio, 0, 4, alignment=Qt.AlignCenter)

        self.btn_infer_vedio = QPushButton("Infer Vedio")
        self.btn_infer_vedio.setFont(QFont("Ubuntu Mono", 7))
        self.btn_infer_vedio.setFixedSize(200, 50)
        self.btn_infer_vedio.clicked.connect(self.btn_infer_vedio_clicked)
        line2.addWidget(self.btn_infer_vedio, 0, 5, alignment=Qt.AlignCenter)

        wgt.setLayout(layout)
        self.win.setCentralWidget(wgt)
        self.win.show()

    def btn_next_pic_clicked(self):
        if self.img_id<len(self.img_list)-1:
            self.img_id += 1
        else:
            self.img_id = 0
        img = self.img_pix[self.img_id]
        self.wait_infer_image = img
        #img = img.scaled(self.label.size(), Qt.KeepAspectRatio)
        img = img.scaled(self.vedio.size(), Qt.KeepAspectRatioByExpanding)
        #self.wait_infer_image = img
        #self.label.setPixmap(img)
        self.vedio.setPixmap(img)

    def btn_slpic_clicked(self):
        imgName, imgType = self.openFile()
        img = QPixmap(imgName)
        self.wait_infer_image = img
        self.temp_image_name = imgName
        #img = img.scaled(self.label.size(), Qt.KeepAspectRatio)
        #self.label.setPixmap(self.wait_infer_image)
        img = img.scaled(self.vedio.size(), Qt.KeepAspectRatioByExpanding)
        self.vedio.setPixmap(self.wait_infer_image)

    def btn_infer_pic_clicked(self):

       pass

    def btn_loc_vedio_clicked(self):
        self.loc_vedio_click = -self.loc_vedio_click
        if self.loc_vedio_click == 1:
            self.startCapture()
        else :
            self.timer.stop()
        
    def btn_slvedio_clicked(self):
        self.sl_vedio_click = -self.sl_vedio_click
        if self.sl_vedio_click == 1:
            vedioName, imgType = self.openFile()
            self.getVedio(vedio_path=vedioName)
        else :
            self.timer.stop()

    def btn_infer_vedio_clicked(self):
        self.infer_vedio_click = -self.infer_vedio_click
        if self.infer_vedio_click == 1:
            self.infer_flag = True
        else :
            self.infer_flag = False
        

    def openFile(self):
        imgName, imgType = QFileDialog.getOpenFileName(self.win, '打开图片', '', '*')
        return imgName, imgType
         

    def startCapture(self):
        #self.setText("请稍候，正在初始化数据和摄像头。。。")
        # 检测相关
        self.cap = cv2.VideoCapture(0)
        self.num_frames = float("inf")
        if not self.cap or not self.cap.isOpened():
            return QMessageBox.critical(self, "错误", "打开摄像头失败")
        self.timer = QTimer(self.win)
        self.timer.timeout.connect(self.onCapture)
        self.timer.stop()
        self.index_frame = 0
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.timer.start(1000 / self.fps)
    
    def getVedio(self, vedio_path):
        self.cap = cv2.VideoCapture(vedio_path)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.cap or not self.cap.isOpened():
            return QMessageBox.critical(self, "错误", "打开摄像头失败")
        self.timer = QTimer(self.win)
        self.timer.timeout.connect(self.onCapture)
        self.timer.stop()
        self.index_frame = 0
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.timer.start(1000 / self.fps)

    def stop(self):
        self.timer.stop()

    def onCapture(self):
        ret, frame = self.cap.read()
        self.index_frame += 1
        if self.index_frame < self.num_frames:
            #minisize = (
            #    int(frame.shape[1] / DOWNSCALE), int(frame.shape[0] / DOWNSCALE))
            #tmpframe = cv2.resize(frame, minisize)
            
            if self.infer_flag:
                
                if self.num_frames < 10000000 and self.index_frame % self.fps_ratio != 0:
                    return True
                else :
                    # single
                    #plt.imshow(frame)
                    #print(frame.shape)
                    #Image.fromarray(frame).save('ori.jpg')
                    pass
                    # m = MLP.MLP()
                    # m.inference(frame)
                    # frame, ans = m.show([0,1], -0.1, 1.2)
                    #print(frame.shape)
                    #Image.fromarray(frame).save('temp.jpg')
                
            else :
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = QImage(
                frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
            del frame
            img = QPixmap(img)
            img = img.scaled(self.vedio.size(), Qt.KeepAspectRatioByExpanding)
            self.vedio.setPixmap(img)    
            return False

if __name__ == "__main__":
    app = QApplication([])
    window = myWindow()
    sys.exit(app.exec_())