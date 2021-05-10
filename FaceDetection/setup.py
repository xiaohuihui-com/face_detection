import sys
import cv2
import numpy as np
from appgui import Ui_MainWindow
from  PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore,QtGui
from zzh_opencv import Demo_opencv

url = "http://admin:admin@192.168.1.49:8081/"
class Demo(QMainWindow,Ui_MainWindow,Demo_opencv):
    def __init__(self):
        super(Demo,self).__init__()
        self.setupUi(self)
        self.cap=cv2.VideoCapture(0 )
        self.solt_init()


    ''' 槽函数初始化 '''
    def solt_init(self):
        self.pushButton.clicked.connect(self.open_video)
        self.pushButton_2.clicked.connect(self.close_video)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_timeout)

    def on_timeout(self):
        if self.cap.isOpened() == True:
            ret,frame=self.cap.read()
            frame = cv2.flip(frame,180) # 水平镜像
            cv2.imwrite('result.jpg',frame)
            detect_result = self.FaceDetect('result.jpg','Cascades/haarcascade_frontalface_alt2.xml')
            if detect_result is not None:
                frame = detect_result

            ''' 将opencv格式图片转化为二进制图片 '''
            img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(img.data,img.shape[1],img.shape[0],img.shape[1]*3,QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.label.setScaledContents(True) # 自适应大小
        else:
            self.cap.release()
    def FaceDetect(self,imgpath, model_path):
        detector = cv2.CascadeClassifier(model_path)
        brg_img = cv2.imread(imgpath)
        gray_img = cv2.cvtColor(brg_img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray_img,
                                          scaleFactor=1.02,
                                          minNeighbors=9,
                                          minSize=(70, 70),
                                          # maxSize=(100, 100),
                                          flags=cv2.CASCADE_SCALE_IMAGE
                                          )
        if len(faces) == 0:
            print('[INFO]: No faces detected...')
            return None
        for (x, y, w, h) in faces:
            cv2.rectangle(brg_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(brg_img, 'Face', (x, y-7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        return brg_img
    def open_video(self):
        self.timer.start(5)
        self.textBrowser.setText('摄像头已打开')

    def close_video(self):
        self.timer.stop()
        self.textBrowser.setText('摄像头已关闭')
        self.cap.release()
        self.label.setPixmap(QtGui.QPixmap("")) # 清除图片

if __name__ == '__main__':
    #此行代码解决designer设计与运行比例不一致
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app= QApplication(sys.argv)
    demo=Demo()
    demo.show()
    sys.exit(app.exec_())
