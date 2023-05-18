from PyQt5.QtWidgets import QApplication, QMainWindow, QStyleFactory
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QBrush, QColor, QPen, QPalette

class CustomTitleBar(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Half and Half Title Bar')
        self.setGeometry(100, 100, 500, 400)

        # set application style to Fusion
        QApplication.setStyle(QStyleFactory.create('Fusion'))

        # set window background color to light gray
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        self.setPalette(palette)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.NoPen))

        # draw first half of title bar
        brush = QBrush(QColor(200, 200, 200))
        painter.setBrush(brush)
        painter.drawRect(0, 0, self.width() // 2, self.style().pixelMetric(QStyleFactory.PM_TitleBarHeight))

        # draw second half of title bar
        brush = QBrush(QColor(100, 100, 100))
        painter.setBrush(brush)
        painter.drawRect(self.width() // 2, 0, self.width() // 2, self.style().pixelMetric(QStyleFactory.PM_TitleBarHeight))

        super().paintEvent(event)

if __name__ == '__main__':
    app = QApplication([])
    window = CustomTitleBar()
    window.show()
    app.exec_()
