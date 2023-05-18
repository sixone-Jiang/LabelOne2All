#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: jyroy
import sys

from PyQt5.QtCore import QUrl
#from PyQt5.QtWebKitWidgets import QWebView
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QListWidget,QStackedWidget
from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHBoxLayout

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QVBoxLayout, QWidget,QMessageBox,QGridLayout,QFileDialog,QStyleFactory
from PyQt5.QtGui import QFont, QPixmap,QImage,QIcon
from PyQt5.QtCore import Qt, QTimer
import sys,cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bin.tools.General import General, StartLabel, Statistics, Coco2Voc, Inference


class LeftTabWidget(QWidget):
    '''左侧选项栏'''
    def __init__(self):
        super(LeftTabWidget, self).__init__()
        self.setObjectName('LeftTabWidget')

        self.setWindowTitle('LeftTabWidget')
        with open('./qss/QListWidgetQSS.qss', 'r') as f:   #导入QListWidget的qss样式
            self.list_style = f.read()

        self.main_layout = QHBoxLayout(self, spacing=0)     #窗口的整体布局
        self.main_layout.setContentsMargins(0,0,0,0)

        self.left_widget = QListWidget()     #左侧选项列表
        self.left_widget.setStyleSheet(self.list_style)
        self.main_layout.addWidget(self.left_widget)

        self.right_widget = QStackedWidget()
        self.main_layout.addWidget(self.right_widget)

        self._setup_ui()

    def _setup_ui(self):
        '''加载界面ui'''

        self.left_widget.currentRowChanged.connect(self.right_widget.setCurrentIndex)   #list和右侧窗口的index对应绑定

        self.left_widget.setFrameShape(QListWidget.NoFrame)    #去掉边框

        self.left_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  #隐藏滚动条
        self.left_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        list_str = ['General','StartLabel','Statistics','Coco2Voc','Inference']
       
        space_tab = '           '
        wgt = [None]*len(list_str)
        for i in range(len(list_str)):
            self.item = QListWidgetItem(space_tab + list_str[i],self.left_widget)   #左侧选项的添加
            self.item.setSizeHint(QSize(30,60))
            wgt[i] = eval(list_str[i])()

            self.right_widget.addWidget(wgt[i])
            #self.item.setTextAlignment(Qt.AlignCenter)                  #居中显示

            #self.browser = QWebView()                                   #右侧用QWebView来显示html网页
            #self.browser.setUrl(QUrl.fromLocalFile('D://python//code//vision//%s' % url_list[i]))
            #self.right_widget.addWidget(self.browser)



def main():
    ''' '''
    app = QApplication(sys.argv)

    main_wnd = LeftTabWidget()
    main_wnd.show()

    app.exec()

if __name__ == '__main__':
    main()