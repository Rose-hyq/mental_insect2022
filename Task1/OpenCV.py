import numpy as np
import cv2 as cv
from pylab import *
import sys

print("start to detect lines...\n")
frame = "Picture.jpg"
img = cv.imread(frame)
cropped_img = img[0:524,0:500]  # 裁剪图像，仅保留纯色背景以便识别轮廓

# 图像二值化
gray = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
# 提取轮廓
contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# RGB图像转HSV图像以便后续颜色分析
HSVimg = cv.cvtColor(cropped_img, cv.COLOR_RGB2HSV)
for cnt in range(len(contours)):

    # 轮廓逼近
    epsilon = 0.01 * cv.arcLength(contours[cnt], True)
    approx = cv.approxPolyDP(contours[cnt], epsilon, True)
    # 分析几何形状
    corners = len(approx)
    shape_type = ""
    if corners == 3:
        # 求解中心位置
        mm = cv.moments(contours[cnt])
        cx = int(mm['m10'] / mm['m00'])
        cy = int(mm['m01'] / mm['m00'])
        # 颜色分析
        color = HSVimg[cy][cx]
        H = color[0]
        # 根据HSV颜色对应表设置阈值，例：紫色125 <= H <= 155
        if H>=125 and H<=155:
            # 提取与绘制轮廓
            cv.circle(cropped_img, (cx, cy), 3, (0, 0, 0), -1)
            cv.drawContours(cropped_img, contours, cnt, (250, 255, 250), 2)
            # 寻找最小外接矩形
            Outpoint = np.array(contours[cnt])
            rect = cv.minAreaRect(Outpoint)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            box = np.array(cv.boxPoints(rect),dtype = int)  # 获取最小外接矩形的4个顶点坐标
            # 绘制最小外接矩形
            for i in range(4):
                cv.line(cropped_img, box[i], box[(i + 1) % 4], (250, 255, 250), 2, 8)

# 展示图像结果
cv.imshow("input image", cropped_img)
cv.waitKey()
cv.destroyAllWindows()
