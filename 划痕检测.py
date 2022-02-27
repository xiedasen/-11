import cv2
from skimage.metrics import structural_similarity
import imutils
import numpy as np
def cutphoto_A(imgA):   # 标准图 裁剪 模块
    gray = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)

    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # x梯度减去y梯度
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # 模糊和阈值图像
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 进行一系列的侵蚀和扩张
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # 计算最大轮廓的旋转包围框
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg = imgA[y1:y1 + hight, x1-150:x1 + width]
    return cropImg

def cutphoto_B(imgB):   # 样本图 裁剪 模块
    gray = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # x梯度减去y梯度
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # 模糊和阈值图像
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 进行一系列的侵蚀和扩张
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # 计算最大轮廓的旋转包围框
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg = imgB[y1:y1 + hight, x1-150:x1 + width]
    return cropImg


def compar_ima(imgA,imgB):
    # 原图灰度转换
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # 计算两个灰度图像之间的结构相似度指数
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    # 找到不同点的轮廓以致于我们可以在被标识为“不同”的区域周围放置矩形
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]

    # 找到一系列区域，在区域周围放置矩形
    flag = 0

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if 200 > w > 1 and 200 > h > 1:    # 矩阵宽度和高度 范围
            cv2.rectangle(imgA, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgB, (x, y), (x + w, y + h), (0, 255, 0), 2)
            flag += 1
        else:
            pass
    return (flag,imgB)

def flag_result(i,img,flag):
    if flag >= 0:
        print("第{}张不合格! ".format(i))
        cv2.imshow('imgaa',img)
        #cv2.imwrite("D:\\requests\\" + str(i) + ".jpg", img)
    else:
        print('第%d张合格!' % i)
    print(flag)
if __name__ == '__main__':
    img1=cv2.imread('15.png')   # 读取标准图
    #img1=cv2.resize(img1,(1000,1000))  # 改变尺寸为 1000*1000

    img2=cv2.imread('16.png')# 读取样本图
    #img2=cv2.resize(img2,(1000,1000))   # 改变尺寸为 1000*1000

    i=1        #第几张的命名
    imgA=cutphoto_A(img1)  # 裁剪标准图 返回裁剪完的标准图（标签）
    imgB=cutphoto_B(img2)   # 裁剪样本图  返回裁剪完的样本图（标签）
    flag,img=compar_ima(imgA,imgB)  # 划痕检测 返回 flag 标记符 和 img3 框取不同点的 样本图
    flag_result(i,img,flag)  # 判断是否合格 并指定地址存放 不合格 的标签
    cv2.waitKey(0)



