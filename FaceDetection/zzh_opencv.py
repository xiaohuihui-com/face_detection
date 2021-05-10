import cv2
import matplotlib.pyplot as plt
import numpy as np


class Demo_opencv():
    def __init__(self):
        pass
        # self.image = cv2.imread(pathImg,mode)
        # self.getImgInfo(self.image)
    ''' 打印图片信息 '''

    def getImgInfo(self, image):
        print("类型：{}".format(type(image)))
        print("图片尺寸：{}".format(image.shape))
        print("图片大小：{}".format(image.size))
        print("图片存储类型：{}".format(image.dtype))

    ''' 图片显示 '''

    def showImg(self, image):
        cv2.imshow("show", image)
        cv2.moveWindow("show", 400, 40)  # 设置窗口位置
        cv2.resizeWindow("show", image.shape[1], image.shape[0]);
        cv2.waitKey(0)  # 等待时间，毫秒级，0表示任意键终止
        cv2.destroyAllWindows()

    ''' 等比例缩放图片 '''

    def resizeImg(self, image, ratio):
        weight = image.shape[0]
        height = image.shape[1]
        img = cv2.resize(image, (int(weight * ratio), int(height * ratio)))
        return img

    ''' 二值化阈值处理 '''

    def Img_threshold(self, imgGray, mode, thresh=127, maxval=255):
        if mode == 1:  # 二值化
            ret, dst = cv2.threshold(imgGray, thresh, maxval, cv2.THRESH_BINARY)
        elif mode == 2:  # 反向二值化
            ret, dst = cv2.threshold(imgGray, thresh, maxval, cv2.THRESH_BINARY_INV)
        elif mode == 3:  # 截断二值化
            ret, dst = cv2.threshold(imgGray, thresh, maxval, cv2.THRESH_TRUNC)
        elif mode == 4:  # 低阈值零处理二值化
            ret, dst = cv2.threshold(imgGray, thresh, maxval, cv2.THRESH_TOZERO)
        elif mode == 5:  # 超阈值零处理二值化
            ret, dst = cv2.threshold(imgGray, thresh, maxval, cv2.THRESH_TOZERO_INV)
        return dst

    ''' 二值化阈值处理 '''

    def Img_threshold(self, imgGray, mode, thresh=127, maxval=255):
        if mode == 1:  # 二值化
            ret, dst = cv2.threshold(imgGray, thresh, maxval, cv2.THRESH_BINARY)
        elif mode == 2:  # 反向二值化
            ret, dst = cv2.threshold(imgGray, thresh, maxval, cv2.THRESH_BINARY_INV)
        elif mode == 3:  # 截断二值化
            ret, dst = cv2.threshold(imgGray, thresh, maxval, cv2.THRESH_TRUNC)
        elif mode == 4:  # 低阈值零处理二值化
            ret, dst = cv2.threshold(imgGray, thresh, maxval, cv2.THRESH_TOZERO)
        elif mode == 5:  # 超阈值零处理二值化
            ret, dst = cv2.threshold(imgGray, thresh, maxval, cv2.THRESH_TOZERO_INV)
        return ret, dst

    ''' 自适应阈值处理 '''

    def Img_adaptiveThreshold(self, imgGray, mode, size=5):
        if mode == 1:
            athd = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, size, 3)
        elif mode == 2:
            athd = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, size, 3)
        elif mode == 3:
            athd = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, size, 3)
        elif mode == 4:
            athd = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, size, 3)
        return athd

    ''' 大率法 '''

    def Img_otsu(self, imgGray, mode):
        if mode == 1:
            t, otsu = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif mode == 2:
            t, otsu = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return otsu

    ''' 滤波处理 '''

    def Img_filter(self, imgGray, mode):
        if mode == 1:  # 均值滤波
            dst = cv2.blur(imgGray, (5, 5))
        elif mode == 2:  # 方框滤波
            dst = cv2.boxFilter(imgGray, -1, (5, 5))
        elif mode == 3:  # 高斯滤波
            dst = cv2.GaussianBlur(imgGray, (5, 5), 0, 0)
        elif mode == 4:  # 中值滤波
            dst = cv2.medianBlur(imgGray, 3)
        elif mode == 5:  # 双边滤波
            dst = cv2.bilateralFilter(imgGray, 25, 100, 100)
        elif mode == 6:  # 2D卷积
            dst = cv2.filter2D(imgGray, -1, 2)
        return dst

    ''' 形态学处理 '''

    def Img_morphology(self, imgGray, mode, times=1):
        kernel = np.ones((5, 5), np.uint8)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形核
        kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字核
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆核
        if mode == 1:  # 腐蚀
            dst = cv2.erode(imgGray, kernel, iterations=times)
        elif mode == 2:  # 膨胀
            dst = cv2.dilate(imgGray, kernel, iterations=times)
        elif mode == 3:  # 开运算
            dst = cv2.morphologyEx(imgGray, cv2.MORPH_OPEN, kernel, iterations=times)
        elif mode == 4:  # 闭运算
            dst = cv2.morphologyEx(imgGray, cv2.MORPH_CLOSE, kernel, iterations=times)
        elif mode == 5:  # 形态学梯度运算
            dst = cv2.morphologyEx(imgGray, cv2.MORPH_GRADIENT, kernel, iterations=times)
        elif mode == 6:  # 礼帽运算
            dst = cv2.morphologyEx(imgGray, cv2.MORPH_TOPHAT, kernel, iterations=times)
        elif mode == 7:  # 黑帽运算
            dst = cv2.morphologyEx(imgGray, cv2.MORPH_BLACKHAT, kernel, iterations=times)
        return dst

    ''' Soble边缘检测 '''

    def Img_Soble(self, imgGray, alpha=0.5, beta=0.5, gamma=10):
        dstx = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0)  # 对x
        dsty = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1)  # 对y
        dstx = cv2.convertScaleAbs(dstx)  # 对运算结果取绝对值
        dsty = cv2.convertScaleAbs(dsty)  # 对运算结果取绝对值
        dstxy = cv2.addWeighted(dstx, alpha, dsty, beta, gamma)
        return dstxy

    ''' Scharr边缘检测 '''

    def Img_Scharr(self, imgGray, alpha=0.5, beta=0.5, gamma=10):
        dstx = cv2.Scharr(imgGray, cv2.CV_64F, 1, 0)  # 对y
        dsty = cv2.Scharr(imgGray, cv2.CV_64F, 0, 1)  # 对y
        dstx = cv2.convertScaleAbs(dstx)  # 对运算结果取绝对值
        dsty = cv2.convertScaleAbs(dsty)  # 对运算结果取绝对值
        dstxy = cv2.addWeighted(dstx, alpha, dsty, beta, gamma)
        return dstxy

    ''' Canny边缘检测 '''

    def Img_Canny(self, imgGray, threshold1=128, threshold2=255):
        dst = cv2.Canny(imgGray, threshold1, threshold2)
        return dst

    def Img_findContours(self, imgage, mode):
        gray = cv2.cvtColor(imgage, cv2.COLOR_BGR2GRAY)
        binary = self.Img_otsu(gray, 2)
        contours, hierachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        temp = np.zeros(imgage.shape, np.uint8)
        if mode == 1:  # 原图画轮廓
            dst = cv2.drawContours(imgage, contours, -1, (0, 0, 255), 5)  # -1代表所有轮廓
        elif mode == 2:  # 轮廓
            dst = cv2.drawContours(temp, contours, 0, (255, 255, 255), 5)
        elif mode == 3:  # 轮廓实心
            dst = cv2.drawContours(temp, contours, 0, (255, 255, 255), -1)
        elif mode == 4:  # 原图前景
            dst = cv2.bitwise_and(imgage, self.Img_findContours(imgage, 3))
        elif mode == 5:  # 原图背景
            dst = cv2.bitwise_or(imgage, self.Img_findContours(imgage, 3))
        return dst

    def videoOpen(self,url=0):
        cap=cv2.VideoCapture(url)
        while(cap.isOpened()):
            ret,frame=cap.read()
            cv2.imshow("frame",frame)
            key=cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('s'):
                cv2.imwrite("../resource/images/save.jpg",frame)
                print('保存成功')
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == '__main__':
    # pathImg = '../resource/images/2.png'
    # image = cv2.imread(pathImg,1)
    cv = Demo_opencv()
    cv.videoOpen()

