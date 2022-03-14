import cv2
import os
import math
from scipy import misc, ndimage
import numpy as np
from numpy.core.fromnumeric import reshape
from sqlalchemy import false, true


# 提取汽油罐和颜色检测
def resize_image(image, height, width):
    top, bottom, left, right = (0, 0, 0, 0)
    # ��ȡͼƬ�ߴ�
    h, w, _ = image.shape
    print('h&w   ', h, '        ', w)

    # ���ڳ����ȵ�ͼƬ���ҵ����һ��
    longest_edge = max(h, w)

    # ����̱���Ҫ���Ӷ������ؿ�Ȳ����볤�ߵȳ�(�൱��padding�����ߵ�paddingΪ0���̱߲Ż���padding)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass  # pass�ǿ���䣬��Ϊ�˱��ֳ���ṹ�������ԡ�pass�����κ����飬һ������ռλ��䡣
    # RGB��ɫ
    BLACK = [0, 0, 0]
    # ��ͼƬ����padding��ʹͼƬ���������
    # top, bottom, left, right�ֱ��Ǹ����߽�Ŀ�ȣ�cv2.BORDER_CONSTANT��һ��border type����ʾ����ͬ����ɫ���
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # ����ͼ���С������ͼ��Ŀ���Ǽ��ټ��������ڴ�ռ�ã�����ѵ���ٶ�
    return cv2.resize(constant, (height, width))


def grab_cut(img, arr):
    grabcutImg = img
    # print('arr   ',arr)
    # ��!!!
    Coords1x, Coords1y, Coords2x, Coords2y = 0, 0, 0, 0
    Coords1x, Coords2x = min(arr[0][0], arr[1][0], arr[2][0], arr[3][0]), max(arr[0][0], arr[1][0], arr[2][0],
                                                                              arr[3][0])
    Coords1y, Coords2y = min(arr[0][1], arr[1][1], arr[2][1], arr[3][1]), max(arr[0][1], arr[1][1], arr[2][1],
                                                                              arr[3][1])

    # ����һ����������ͼ��ͬ��״��ͼ����ģ,ȡֵ��0,1,2,3
    mask = np.zeros(img.shape[:2], np.uint8)

    # ����ģʽ,����Ϊ1��,13x5��
    bgModel = np.zeros((1, 65), np.float64)
    # ǰ��ģʽ,����Ϊ1��,13x5��
    fgModel = np.zeros((1, 65), np.float64)
    if (Coords2x - Coords1x) > 0 and (Coords2y - Coords1y) > 0:
        # �ָ�ľ�������
        # rect = (Coords1x, Coords1y, Coords2y - Coords1y, Coords2x - Coords1x)
        rect = (Coords1x, Coords1y, Coords2x, Coords2y)
        # print('#### fenge quyu:',rect)
        # print("###########")
        iterCount = 17
        # grabCut����,GC_INIT_WITH_RECTģʽ
        cv2.grabCut(img, mask, rect, bgModel, fgModel, iterCount, cv2.GC_INIT_WITH_RECT)
        # grabCut����,GC_INIT_WITH_MASKģʽ
        cv2.grabCut(img, mask, rect, bgModel, fgModel, iterCount, cv2.GC_INIT_WITH_MASK)
        # ������0,2���0,�������1
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        # ���¼���ͼ����ɫ,��ӦԪ�����
        grabcutImg = img * mask2[:, :, np.newaxis]

    # cv2.drawContours(srcimage, [points], -1, (255,0,0), 2)
    else:
        print('########test#########', Coords2x - Coords1x, Coords2y - Coords1y)
    return grabcutImg, rect


# cv2.waitKey(0)

def diffImg(srcImg, gbImg):
    diffImg = cv2.subtract(srcImg, gbImg)
    return diffImg


def img2Alpha(crop_image):
    tmp = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(crop_image)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    # ע�Ᵽ���png��ʽ,jpg�Ļ����Ǻ�ɫ����(255)
    return dst


def getLoc(img_init, arr1, arr2):
    img_hsv = cv2.cvtColor(img_init, cv2.COLOR_BGR2HSV)
    # thresh1,thresh2=np.array([8,5,0]),np.array([185,171,132])
    thresh1, thresh2 = arr1, arr2
    img_rng = cv2.inRange(img_hsv, thresh1, thresh2)
    # cv2.imshow("img_rng",img_rng)
    img_morph = img_rng.copy()
    # ��ʴ
    cv2.erode(img_morph, (3, 3), img_morph, iterations=3)
    # ����
    cv2.dilate(img_morph, (3, 3), img_morph, iterations=3)

    # ��ȡͼ������
    # ��ȡͼ������
    cnts, _ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # ����������Ӵ�С����
    cnts_sort = sorted(cnts, key=cv2.contourArea, reverse=True)
    # ѡȡ�����������������ó���С��Ӿ���
    box = cv2.minAreaRect(cnts_sort[0])
    points = np.int0(cv2.boxPoints(box))
    # print('point       ',points)
    # cv2.drawContours(img_init, [points], -1, (255,0,0), 2)
    # cv2.imwrite('E:/MyProject/pythonTest/py/CV/test', img_init)
    return points


def getHSVcount(image, rect):
    sum, count = 0, 0
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # print(w,'########',h)

    for y in range(rect[1], rect[3]):
        for x in range(rect[0], rect[2]):
            sum += 1
            if 20 < HSV[x, y][0] < 30:
                # sum+= HSV[x,y][0]
                count += 1
            else:
                pass
    if count == 0:
        count = 1
    # print(sum,'##############',count)
    return count / sum


def getRes(per):
    if 0.2 < per < 0.8:
        return True
    else:
        return False

def read__image(image_name):
    # i = 0
    srcimage = cv2.imread(image_name)
    image_rs=cv2.resize(srcimage,(400,400))
    points=getLoc(image_rs,np.array([8,5,0]),np.array([185,171,132]))
    gbimage,rect=grab_cut(image_rs,points)
    cv2.imwrite('grabcut.png',gbimage)
    tmp = cv2.cvtColor(gbimage, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(gbimage)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    # cv2.imshow('dst',dst)

    percent=getHSVcount(dst,rect)
    # print(name,'       percent:=     ',percent)
    res=getRes(percent)
    if res==True:
        print(image_name,',颜色合格')
    else:
        print(image_name,',颜色不合格')
    # cv2.waitKey(0)
    #�����յ�ͼƬ��������
    # ע�Ᵽ���png��ʽ,jpg�Ļ����Ǻ�ɫ����(255)
    # image_name = '%s%s.png' % ('CV/output/',name)  # ע������ͼƬ��һ��Ҫ������չ�����������imwrite��ʱ��ᱨ��
    # print(image_name)
    # cv2.imwrite(image_name, dst)


#提取标签
def regrab_cut(source):
    # ��ȡͼƬ
    # img = cv2.imread(sourceDir)
    img = source

    # ͼƬ���
    # print(img.shape)
    # ͼƬ�߶�
    #  = img.shape[0]
    # �ָ�ľ�������
    y,x,_= img.shape
    rect = (3,3,x-80,y-60)
    # ����ģʽ,����Ϊ1��,13x5��
    bgModel = np.zeros((1, 65), np.float64)
    # ǰ��ģʽ,����Ϊ1��,13x5��
    fgModel = np.zeros((1, 65), np.float64)
    # ͼ����ģ,ȡֵ��0,1,2,3
    mask = np.zeros(img.shape[:2], np.uint8)
    # grabCut����,GC_INIT_WITH_RECTģʽ
    cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    # grabCut����,GC_INIT_WITH_MASKģʽ
    cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_MASK)
    # ������0,2���0,�������1
    mask2 = np.where((mask == 3) | (mask == 1), 0, 1).astype('uint8')
    # ���¼���ͼ����ɫ,��ӦԪ�����
    grabcutImg1 = img * mask2[:, :, np.newaxis]
    # cv2.imwrite('test5.png', img2)

    return grabcutImg1

# cv2.waitKey(0)

#检测标签是否倾斜
# 通过霍夫变换计算角度
def CalcDegree(srcImage):
    midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    dstImage = cv2.Canny(midImage, 5, 250, 3)
    lineimage = srcImage.copy()

    # 通过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高
    lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 100)
    # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    sum = 0
    # 依次画出每条线段
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            # print("theta:", theta, " rho:", rho)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))
            # 只选角度最小的作为旋转角度
            sum += theta
            cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imwrite("Imagelines.png", lineimage)

    # 对所有角度求平均，这样做旋转效果会更好
    average = sum / len(lines)

    if average > 1.595 or average<1.381:
        print("角度不合格")
    else:
        print("角度合格")
    return average

#矫正标签
def rotate(image, angle, center=None, scale=1.0):
    (w, h) = image.shape[0:2]
    if center is None:
        center = (w // 2, h // 2)
    wrapMat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, wrapMat, (h, w))

# 使用霍夫变换
def getCorrect():
    average = CalcDegree(Image)
    if average > 1.595 or average < 1.381:
        Img0 = "small.png"
        # 读取图片，灰度化
        src = cv2.imread(Img0, cv2.IMREAD_COLOR)

        gray = cv2.imread(Img0, cv2.IMREAD_GRAYSCALE)

        # 腐蚀、膨胀
        kernel = np.ones((5, 5), np.uint8)
        erode_Img = cv2.erode(gray, kernel)
        eroDil = cv2.dilate(erode_Img, kernel)

        # 边缘检测
        canny = cv2.Canny(eroDil, 10, 150)
        cv2.imwrite("Canny.png", canny)
        # 霍夫变换得到线条
        lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        drawing = np.zeros(src.shape[:], dtype=np.uint8)
        # 画出线条
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)

        cv2.imwrite("houghP.png", drawing)
        """
        计算角度,因为x轴向右，y轴向下，所有计算的斜率是常规下斜率的相反数，我们就用这个斜率（旋转角度）进行旋转
        """
        k = float(y1 - y2) / (x1 - x2)
        thera = np.degrees(math.atan(k))

        """
        旋转角度大于0，则逆时针旋转，否则顺时针旋转
        """
        rotateImg = rotate(src, thera)
    else:
        rotateImg = cropped

    return rotateImg

if __name__=='__main__':
    #输入图像进行罐体提取和颜色检测
    read__image('frames_0.jpg')
    #提取标签
    sourceDir = "grabcut.png"
    source = cv2.imread(sourceDir)
    bgImg = regrab_cut(source)
    srcImg = cv2.imread(sourceDir)
    diffImg1 = cv2.subtract(srcImg, bgImg)
    cv2.imwrite('diffImg1.png', diffImg1)
    #裁剪图片
    imgA = cv2.imread('diffImg1.png')
    cropped = imgA[50:350, 50:350]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite("small.png", cropped)
    #检测图像是否倾斜
    input_img_file = "small.png"
    Image = cv2.imread(input_img_file)
    average = CalcDegree(Image)
    #矫正标签
    rotateImg = getCorrect()
    cv2.imwrite("rotateImg.png", rotateImg)


