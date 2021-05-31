import time
from detect import *

if __name__ == '__main__':

    start_time = time.time()

    image = cv2.imread("./test_images/1.jpg")
    Cascade_Image = Cascade_Dection(image)

    flag = 0  # flag的作用是用来检测如果Cascade级联分类器初步分类到车牌但是后期HSV却无法识别。往往这种情况是由于图像过于模糊情况下，当然这也是级联分类器优点

    if len(Cascade_Image) == 0:  # 这里是判断如果刚开始的级联分类器识别不到车牌就要启用HSV识别
        print("Cascade级联分类器未检测到车牌进入HSV识别模式")
        img = HSV_Detection(image)  # HSV识别
        ctrs, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找出轮廓
        for k in range(len(ctrs)):  # 遍历所有轮廓
            c = sorted(ctrs, key=cv2.contourArea, reverse=True)[k]
            rect = cv2.minAreaRect(c)  # 在轮廓上画出包围轮廓的最小矩形
            box = np.int0(cv2.boxPoints(rect))  # 得到矩形的四个点坐标，放在box里面
            Xs = [k[0] for k in box]
            Ys = [k[1] for k in box]
            x1 = abs(min(Xs))
            x2 = abs(max(Xs))
            y1 = abs(min(Ys))
            y2 = abs(max(Ys))
            height = y2 - y1  # 得到矩形宽度
            width = x2 - x1  # 得到矩形高度
            if width / height > 4 or width / height < 1 or cv2.contourArea(box) < 800:  # 判断车牌的条件，如果比例和面积不符合就继续循环下一个
                continue
            flag = 1
            Crop = image[int(y1 * 0.9):y1 + int(height * 1.2),
                   int(x1 * 0.9):x1 + int(width * 1.2)]  # 抠出包含矩形的一个CROP区域顺便尽可能扩大化区域防止信息丢失
            compare_value = box[0][0] - box[2][0]  # 这是判断左角度看到还是右角度看到的车牌
            # cv2.drawContours(image,[box], -1, (0, 0, 255), 1)
            Crop1 = HSV(Crop)  # 对CROP车牌再一次进行HSV识别更精确的框出

            ctrs0, hier0 = cv2.findContours(Crop1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 下面和上述步骤差不多
            for m in range(len(ctrs0)):
                c0 = sorted(ctrs0, key=cv2.contourArea, reverse=True)[m]
                rect0 = cv2.minAreaRect(c0)
                box0 = np.int0(cv2.boxPoints(rect0))
                Xs0 = [m[0] for m in box0]
                Ys0 = [m[1] for m in box0]
                x10 = abs(min(Xs0))
                x20 = abs(max(Xs0))
                y10 = abs(min(Ys0))
                y20 = abs(max(Ys0))
                height0 = y20 - y10
                width0 = x20 - x10
                if width0 / height0 > 4 or width0 / height0 < 1 or cv2.contourArea(box) < 800:
                    continue
                # cv2.drawContours(Crop, [box0], -1, (0, 0, 255), 1)
                # cv2.imshow("ww", Crop)

                if compare_value < 0:  # 这是从左角度看到的车牌
                    pltPoint = np.float32([box0[1], box0[2], box0[0], box0[3]])  # 车牌透视变换前的四个点打包成矩阵
                    dstPoint = np.float32([[0, 0], [180, 0], [0, 60], [180, 60]])  # 车牌透视变换后的四个点
                    M = cv2.getPerspectiveTransform(pltPoint, dstPoint)  # 得到变换矩阵
                    Crop0 = cv2.warpPerspective(Crop, M, (180, 60))  # 得到透视变换后的图像
                    res, confidence = recognizeOne(Crop0)  # 识别车牌信息
                    print("车牌号： %s 置信度： %.4f" % (res, confidence))
                    r = [box[1][0], box[2][1], width, height]
                    image = drawRectBox(image, r, res + " " + str(round(confidence, 3)))
                    cv2.imshow("wd", Crop0)
                    cv2.imshow("OR", image)
                if compare_value > 0:
                    pltPoint = np.float32([box0[2], box0[3], box0[1], box0[0]])
                    dstPoint = np.float32([[0, 0], [180, 0], [0, 60], [180, 60]])
                    M = cv2.getPerspectiveTransform(pltPoint, dstPoint)
                    Crop0 = cv2.warpPerspective(Crop, M, (180, 60))
                    res, confidence = recognizeOne(Crop0)
                    r = [box[2][0], box[2][1], width, height]
                    image = drawRectBox(image, r, res + " " + str(round(confidence, 3)))
                    print("车牌号： %s 置信度： %.4f" % (res, confidence))
                    cv2.imshow("wd", Crop0)
                    cv2.imshow("ORI", image)

    for j, plate in enumerate(Cascade_Image):  # 对初步筛选的车牌进行遍历
        print("已进入Cascade检测")
        plate, rect = plate
        img1 = HSV_Detection(plate)  # 对筛选到的车牌进行HSV框选出更好更准确的位置
        ctrs1, hier1 = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找边界
        for i in range(len(ctrs1)):
            c1 = sorted(ctrs1, key=cv2.contourArea, reverse=True)[i]
            rect1 = cv2.minAreaRect(c1)  # 画出包围车牌的最小矩形
            box1 = np.int0(cv2.boxPoints(rect1))  # 得到矩形的四个位置点
            Xs1 = [i[0] for i in box1]
            Ys1 = [i[1] for i in box1]
            x1 = abs(min(Xs1))
            x2 = abs(max(Xs1))
            y1 = abs(min(Ys1))
            y2 = abs(max(Ys1))
            height1 = y2 - y1  # 得到宽度
            width1 = x2 - x1  # 得到高度
            if width1 / height1 > 4 or width1 / height1 < 1.5 or cv2.contourArea(box1) < 800:  # 如果不满足车牌的条件不进行下面步骤
                continue
            flag = 2
            # cv2.drawContours(plate, [box1], -1, (0, 0, 255), 1)       #用红色线框出车牌
            # cv2.imshow("plate",plate)
            Crop_Image1 = plate[y1:y1 + height1, x1:x1 + width1]  # 提取出第一次定位到的车牌
            compare_value = box1[0][0] - box1[2][0]  # 因为车牌的拍摄总共有两个角度左边和右边

            if compare_value > 0:  # 第一种车牌倾斜情况
                pltPoint = np.float32([box1[2], box1[3], box1[1], box1[0]])  # 车牌变换前的四个点坐标，分别为左上，右上，左下，右下
                dstPoint = np.float32([[0, 0], [180, 0], [0, 60], [180, 60]])  # 车牌变换后的四个点坐标，还是对应左上，右上，左下，右下
                M = cv2.getPerspectiveTransform(pltPoint, dstPoint)  # 得到变换矩阵Mat
                Crop_Image2 = cv2.warpPerspective(plate, M, (440, 140))  # 进行矩阵相乘得到透视变换后的图像
                Crop_Image3 = cv2.bitwise_or(HSV(Crop_Image2),
                                             HSV_Detection(Crop_Image2))  # 透视变换后要再进行一次HSV定位因为有些地方会被插补掉
                # cv2.imshow("CROPI", Crop_Image2)
                # cv2.imshow("CROPII", Crop_Image3)
                cts2, hier2 = cv2.findContours(Crop_Image3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 再重复之前动作找边界
                for j in range(len(cts2)):
                    c2 = sorted(cts2, key=cv2.contourArea, reverse=True)[j]
                    rect2 = cv2.minAreaRect(c2)  # 找出透视变换后车牌的最小矩形
                    box2 = np.int0(cv2.boxPoints(rect2))
                    Xs2 = [j[0] for j in box2]
                    Ys2 = [j[1] for j in box2]
                    a1 = abs(min(Xs2))
                    a2 = abs(max(Xs2))
                    b1 = abs(min(Ys2))
                    b2 = abs(max(Ys2))
                    height2 = b2 - b1
                    width2 = a2 - a1
                    # cv2.drawContours(Crop_Image2, [box2], -1, (0, 0, 255), 1)
                    # cv2.imshow("CROPI", Crop_Image2)
                    CROP = Crop_Image2[b1:b1 + height2, a1:a1 + width2]  # 选出最后矫正后的车牌的位置
                    cv2.imshow("DST", CROP)
                res, confidence = recognizeOne(CROP)  # 识别车牌内容
                image = drawRectBox(image, rect, res + " " + str(round(confidence, 3)))
                cv2.imshow("ORI", image)
                print("车牌号： %s 置信度： %.4f" % (res, confidence))

            if compare_value < 0:  # 第二种车牌倾斜情况，接下去的操作和上述一样
                pltPoint = np.float32([box1[1], box1[2], box1[0], box1[3]])
                dstPoint = np.float32([[0, 0], [180, 0], [0, 60], [180, 60]])
                M = cv2.getPerspectiveTransform(pltPoint, dstPoint)
                Crop_Image2 = cv2.warpPerspective(plate, M, (440, 140))
                Crop_Image3 = cv2.bitwise_or(HSV(Crop_Image2), HSV_Detection(Crop_Image2))

                cts2, hier2 = cv2.findContours(Crop_Image3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for j in range(len(cts2)):
                    c2 = sorted(cts2, key=cv2.contourArea, reverse=True)[j]
                    rect2 = cv2.minAreaRect(c2)
                    box2 = np.int0(cv2.boxPoints(rect2))
                    Xs2 = [j[0] for j in box2]
                    Ys2 = [j[1] for j in box2]
                    a1 = abs(min(Xs2))
                    a2 = abs(max(Xs2))
                    b1 = abs(min(Ys2))
                    b2 = abs(max(Ys2))
                    height2 = b2 - b1
                    width2 = a2 - a1
                    CROP = Crop_Image2[b1:b1 + height2, a1:a1 + width2]
                    cv2.imshow("DST", CROP)
                res, confidence = recognizeOne(CROP)
                image = drawRectBox(image, rect, res + " " + str(round(confidence, 3)))
                cv2.imshow("ORI", image)
                print("车牌号： %s 置信度： %.4f" % (res, confidence))
        break

    if flag == 0:
        print("图像模糊")
        Image = Normal_Cascade(image)
        for j, plate in enumerate(Image):
            plate, rect = plate
            image_rgb, rect_refine = finemappingVertical(plate, rect)
            res, confidence = recognizeOne(image_rgb)
            image = drawRectBox(image, rect_refine, res + " " + str(round(confidence, 3)))
            print(res, confidence)
        cv2.imshow("DST", image)

    end_time = time.time()
    print("总共用时: %.4f" % (end_time - start_time))

    cv2.waitKey(0)
