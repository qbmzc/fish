import cv2
import numpy as np
import os


def segmentation(image,r):
    "预处理"
    # src = cv2.resize(image,None,fx=0.3,fy=0.3)    # 缩小大小
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)     # 灰度化
    #自适应阈值二值化
    blocksize = 9
    C = 4
    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, C)
    blur0 = cv2.GaussianBlur(binary,(9,9),0)     # 去噪
    blur1 = cv2.cvtColor(blur0,cv2.COLOR_GRAY2BGR)

    "鱼体和硬币分割"
    # # 得到锚框
    # r = cv2.selectROI('input', blur1, False)  # 返回 (x_min, y_min, w, h)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # roi区域
    # roi = src[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # 原图mask
    mask = np.zeros(src.shape[:2], dtype=np.uint8)

    # 矩形roi
    rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3])) # 包括前景的矩形，格式为(x,y,w,h)
    # rect = (1, 1, image.shape[1], image.shape[0])

    bgdmodel = np.zeros((1,65),np.float64) # bg模型的临时数组
    fgdmodel = np.zeros((1,65),np.float64) # fg模型的临时数组

    cv2.grabCut(blur1,mask,rect,bgdmodel,fgdmodel, 5, mode=cv2.GC_INIT_WITH_RECT)

    # 提取前景和可能的前景区域
    mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
    # mask2 = np.where((mask==1), 255, 0).astype('uint8')

    result = cv2.bitwise_and(src,src,mask=mask2)
    # cv2.imwrite('result.jpg', result)
    # cv2.imwrite('roi.jpg', roi)

    return result,blur1
def get_items(image,seg_image):
    "分割结果处理，图像形状特征提取"

    result_gray = cv2.cvtColor(seg_image,cv2.COLOR_BGR2GRAY)     # 灰度化
    #自适应阈值
    blocksize = 7
    C = -5
    binary = cv2.adaptiveThreshold(result_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, C)
    # cv2.imshow("binary", binary)
    blur0 = cv2.GaussianBlur(binary,(15,15),0)
    # cv2.imshow("blur0", blur0)
    blur = cv2.medianBlur(blur0,11)
    # cv2.imshow("blur", blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 找到轮廓
    contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    # 注意要将原图(彩色图)复制一份，不然原图会在函数处理完之后改变
    draw = image.copy()  #draw改变不会导致img改变
    draw1 = image.copy()

    # 获取某一个轮廓用于计算
    area_list = []
    for cnt in contours:
        # 面积
        area = cv2.contourArea(cnt)
        area_list.append(area)
    area_list = np.array(area_list)
    index = area_list.argsort()     # 按面积从小到大排序的索引,倒数第一是鱼，倒数第二是硬币
    if len(index) < 2:
        print("don't find coin,please try another image")
        return (0,0,0,0),None
    # 在draw画板上绘制轮廓；画第几个轮廓，-1代表所有轮廓；BGR分别对应(0,0,255)，用红色线画轮廓；线条宽2
    # res1 = cv2.drawContours(draw, contours,index[-2], (0, 0, 255), 2)
    res1 = cv2.drawContours(draw, contours,index[-1], (0, 0, 255), 2)
    # cv2.imshow('res1',res1)
    # 硬币轮廓的外接矩形
    cnt = contours[index[-2]]
    x,y,w,h = cv2.boundingRect(cnt)  # 计算外接矩形，返回矩形的左上坐标点，和一长一宽
    rectangle = cv2.rectangle(draw,(x,y),(x+w,y+h),(0,255,0),2)  # 根据坐标绘制矩形，绿色线条

    scale = 25 / min(w,h)

    # 鱼轮廓的外接矩形
    cnt_fish = contours[index[-1]]
    x1,y1,w1,h1 = cv2.boundingRect(cnt_fish)  # 计算外接矩形，返回矩形的左上坐标点，和一长一宽
    rectangle_fish = cv2.rectangle(draw,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)  # 根据坐标绘制矩形，绿色线条
    # cv2.imshow('rectangle_fish',rectangle_fish)
    # cv2.imshow('seg_image', seg_image)

    girth = cv2.arcLength(cnt_fish,True)    # 周长
    area_fish = area_list[index[-1]]    # 面积

    w1_real = w1 * scale
    h1_real = h1 * scale
    girth_real = girth * scale
    area_real = area_fish * scale * scale

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return (w1_real,h1_real,girth_real,area_real),rectangle_fish

"end"


if __name__ == '__main__':
    path = "./Data/03"
    img_name = os.listdir(path)[0]
    src = cv2.imread(os.path.join(path, img_name))
    src = cv2.resize(src, None, fx=0.3, fy=0.3) # 缩小大小
    # 得到锚框
    r = cv2.selectROI('input', src, False)  # 返回 (x_min, y_min, w, h)

    "获取图像分割结果"
    # segmentation 需要识别区域的坐标，目前是手工实现的，看前端解决还是怎么处理
    result,blur = segmentation(src,r)    # result是分割的结果图,blur是预处理结果
    "获取识别结果"
    # 如果(w, g, girth, area)四个参数都是0，说明没侦测到硬币，做出提示。
    # draw_img是画了周长、框的图
    (w, g, girth, area),draw_img = get_items(src,result)

    cv2.imshow('result', result)
    cv2.imshow('blur', blur)
    cv2.imshow('draw_img', draw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()