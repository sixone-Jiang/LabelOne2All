from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QWidget, QLabel, QInputDialog
import cv2, os
import xml.etree.ElementTree as ET

class DrawRect(QLabel):
    def __init__(self, parent=None):
        super(DrawRect, self).__init__(parent)
        self.resize(600, 400)
        self.start_point = None
        self.end_point = None
        self.start = False
        self.drawing = False
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    def paintEvent(self, event):
        super(DrawRect, self).paintEvent(event)
        if self.drawing:
            painter = QPainter()
            painter.begin(self)
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            painter.drawRect(QRect(self.start_point, self.end_point))
            #painter.end()

    def mousePressEvent(self, event):
        if not self.start:
            self.start_point = event.pos()
            self.start = True

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.start = False

    def mouseMoveEvent(self, event):
        if self.start:
            self.end_point = event.pos()
            self.drawing = True
            self.update()

class myDrawRect(QLabel):
    def __init__(self, parent=None):
        super(myDrawRect, self).__init__(parent)
        self.start_point = []
        self.end_point = []
        self.label_list = []
        self.filter_list = []
        self.click_flag_left = False
        self.click_flag_right = False
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.now_point = QPoint(0, 0)

    def paintEvent(self, event):
        super(myDrawRect, self).paintEvent(event)
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)

        for start_point, end_point in zip(self.start_point, self.end_point):
            if self.filter_list == []:
                painter.drawRect(QRect(start_point, end_point))
                if self.click_flag_left == False and self.start_point.index(start_point) < len(self.label_list):
                    painter.drawText(start_point, self.label_list[self.start_point.index(start_point)])

            elif self.filter_list != []:
                if self.click_flag_left == True:
                    painter.drawRect(QRect(start_point, end_point))
                elif self.click_flag_left == False and self.start_point.index(start_point) < len(self.label_list):
                    if self.label_list[self.start_point.index(start_point)] in self.filter_list:
                        painter.drawRect(QRect(start_point, end_point))
                        painter.drawText(start_point, self.label_list[self.start_point.index(start_point)])

        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point.append(event.pos())
            self.end_point.append(event.pos())
            self.click_flag_left = True
        elif event.button() == Qt.RightButton:
            index = self.match_rect(event.pos())
            self.click_flag_right = True
            if index != -1:
                self.start_point.pop(index)
                self.end_point.pop(index)
                self.label_list.pop(index)
            self.update()

    def del_rect(self):
        index = -1
        self.start_point.pop(index)
        self.end_point.pop(index)
        self.label_list.pop(index)
        self.update()

    def mouseReleaseEvent(self, event):
        if self.click_flag_left:
            #self.end_point.append(event.pos())
            wgt_temp = QWidget()
            self.click_flag_left = False
            text, ok = QInputDialog.getText(wgt_temp, 'info', '请输入标签：')
            if ok and text:
                self.label_list.append(str(text))
            else :
                self.label_list.append('None')
            #self.update()

        elif self.click_flag_right:
            self.click_flag_right = False

    def mouseMoveEvent(self, event):
        self.now_point = event.pos()
        if self.click_flag_left:
            self.end_point[-1] = event.pos()
            self.update()

    # 通过一个点的坐标匹配最合适的矩形框
    def match_rect(self, point):
        for i in range(len(self.start_point)):
            if self.start_point[i].x() <= point.x() <= self.end_point[i].x() and self.start_point[i].y() <= point.y() <= self.end_point[i].y():
                return i
        return -1
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

    # convert to voc format
    def convert_voc(self, img_path, xml_path):
        img = cv2.imread(img_path)
        height, width, depth = img.shape
        with open(xml_path, 'w') as f:
            f.write(self.headstr % (os.path.basename(img_path), width, height, depth))
            for i in range(len(self.start_point)):
                xmin = self.start_point[i].x()
                ymin = self.start_point[i].y()
                xmax = self.end_point[i].x()
                ymax = self.end_point[i].y()
                f.write(self.objstr % (self.label_list[i], xmin, ymin, xmax, ymax))
            f.write('</annotation>')
    
    def from_voc_init(self, xml_path):
        self.start_point = []
        self.end_point = []
        self.label_list = []
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
                self.start_point.append(QPoint(xmin, ymin))
                self.end_point.append(QPoint(xmax, ymax))
                self.label_list.append(name)
        print(self.start_point, self.end_point, self.label_list)
        self.update()
    
    def re_init(self):
        self.start_point = []
        self.end_point = []
        self.label_list = []
        self.update()
    
    def set_filter(self, filter_list):
        self.filter_list = filter_list
        self.update()

    def get_filter(self):
        return str(self.filter_list)