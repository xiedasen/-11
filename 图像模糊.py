import cv2
from PIL import Image, ImageDraw, ImageFont

# 以清晰图片的值作为阈值threshold
def ImgText():
    image = cv2.imread("D:\\Python\\photo\\13.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm1 = cv2.Laplacian(gray, cv2.CV_64F).var()
    threshold = fm1

    img = cv2.imread("D:\\Python\\photo\\14.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # BGR和灰度图的转换使用cv2.COLOR_BGR2GRAY   返回值：颜色空间转换后的图片矩阵
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()     # 拉普拉斯算子
    return (fm,threshold)
if __name__ == "__main__" :
    fm,threshold=ImgText()
    print('image vague is {}'.format(fm))
    print(threshold)

    if threshold <= fm <= 1.05*threshold:
        print('合格!')
    else:
        print('不合格！')