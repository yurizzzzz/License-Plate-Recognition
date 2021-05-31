import sys
import cv2
import importlib
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from keras.models import *
from keras.layers import *

chars = [
    u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂",
    u"湘", u"粤", u"桂",
    u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7",
    u"8", u"9", u"A",
    u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U",
    u"V", u"W", u"X",
    u"Y", u"Z", u"港", u"学", u"使", u"警", u"澳", u"挂", u"军", u"北", u"南", u"广", u"沈", u"兰", u"成", u"济", u"海", u"民",
    u"航", u"空"
]


def fastdecode(y_pred):
    results = ""
    confidence = 0.0
    table_pred = y_pred.reshape(-1, len(chars) + 1)
    res = table_pred.argmax(axis=1)
    for i, one in enumerate(res):
        if one < len(chars) and (i == 0 or (one != res[i - 1])):
            results += chars[one]
            confidence += table_pred[i][one]
    print(confidence, len(results))
    confidence /= len(results)

    return results, confidence


def model_seq_rec(model_path):
    width, height, n_len, n_class = 164, 48, 7, len(chars) + 1
    rnn_size = 256
    input_tensor = Input((164, 48, 3))
    x = input_tensor
    base_conv = 32
    for i in range(3):
        x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)
    x = concatenate([gru_2, gru_2b])
    x = Dropout(0.25)(x)
    x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)
    base_model.load_weights(model_path)
    return base_model


modelSeqRec = model_seq_rec("./model/ocr_plate_all_gru.h5")


def recognizeOne(src):
    x_tempx = src
    x_temp = cv2.resize(x_tempx, (164, 48))
    x_temp = x_temp.transpose(1, 0, 2)
    y_pred = modelSeqRec.predict(np.array([x_temp]))
    y_pred = y_pred[:, 2:, :]
    return fastdecode(y_pred)


def HSV(src):
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    HSV_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    Blue_image = cv2.inRange(HSV_image, lower_blue, upper_blue)
    Output = cv2.bitwise_not(Blue_image, Blue_image)
    Output = cv2.bitwise_not(Output, Output)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    Output = cv2.morphologyEx(Output, cv2.MORPH_CLOSE, kernel)
    Output = cv2.erode(Output, None, iterations=4)
    return Output


def HSV_Detection(src):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_tophat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)
    img_blackhat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
    image_plus_tophat = cv2.add(src, img_tophat)
    image_plus_blackhat_minus_blackhat = cv2.subtract(image_plus_tophat, img_blackhat)

    lower_blue = np.array([60, 130, 90])
    upper_blue = np.array([160, 255, 200])
    HSV_image = cv2.cvtColor(image_plus_blackhat_minus_blackhat, cv2.COLOR_BGR2HSV)
    Blue_image = cv2.inRange(HSV_image, lower_blue, upper_blue)
    Output = cv2.bitwise_not(Blue_image, Blue_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dst = cv2.erode(Output, None, iterations=4)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    dst = cv2.bitwise_not(dst, dst)

    return dst


def computeSafeRegion(shape, bounding_rect):
    top = bounding_rect[1]
    bottom = bounding_rect[1] + bounding_rect[3]
    left = bounding_rect[0]
    right = bounding_rect[0] + bounding_rect[2]
    min_top = 0
    max_bottom = shape[0]
    min_left = 0
    max_right = shape[1]
    if top < min_top:
        top = min_top
    if left < min_left:
        left = min_left
    if bottom > max_bottom:
        bottom = max_bottom
    if right > max_right:
        right = max_right
    return [left, top, right - left, bottom - top]


def Nomal_cropImage(image, rect):
    x, y, w, h = computeSafeRegion(image.shape, rect)
    return image[round(y):y + round(h), round(x):x + round(w)]


def cropImage(image, rect):
    x, y, w, h = computeSafeRegion(image.shape, rect)
    return image[round(y * 0.83):y + round(h * 1.5), round(x * 0.85):x + round(w * 1.2)]


def Normal_Cascade(src):
    watch_cascade = cv2.CascadeClassifier("./model/cascade.xml")
    height = src.shape[0]
    padding = int(height * 0.1)
    scale = src.shape[1] / float(src.shape[0])
    image = cv2.resize(src, (int(scale * src.shape[0]), src.shape[0]))
    image_color_cropped = image[padding:src.shape[0] - padding, 0:src.shape[1]]
    Gray_Image = cv2.cvtColor(image_color_cropped, cv2.COLOR_RGB2GRAY)
    watches = watch_cascade.detectMultiScale(Gray_Image, 1.08, 2, minSize=(36, 9), maxSize=(36 * 40, 9 * 40))
    cropped_images = []
    for (x, y, w, h) in watches:
        x -= w * 0.14
        w += w * 0.28
        y -= h * 0.15
        h += h * 0.3
        cropped = Nomal_cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
        # cv2.imshow("CROP",cropped)
        cropped_images.append([cropped, [x, y + padding, w, h]])
    return cropped_images


def Cascade_Dection(src):
    watch_cascade = cv2.CascadeClassifier("./model/cascade.xml")
    height = src.shape[0]
    padding = int(height * 0.1)
    scale = src.shape[1] / float(src.shape[0])
    image = cv2.resize(src, (int(scale * src.shape[0]), src.shape[0]))
    image_color_cropped = image[padding:src.shape[0] - padding, 0:src.shape[1]]
    Gray_Image = cv2.cvtColor(image_color_cropped, cv2.COLOR_RGB2GRAY)
    watches = watch_cascade.detectMultiScale(Gray_Image, 1.08, 2, minSize=(36, 9), maxSize=(36 * 40, 9 * 40))
    cropped_images = []
    for (x, y, w, h) in watches:
        x -= w * 0.14
        w += w * 0.28
        y -= h * 0.15
        h += h * 0.3
        cropped = cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
        # cv2.imshow("CROP",cropped)
        cropped_images.append([cropped, [x, y + padding, w, h]])
    return cropped_images


def model_finemapping():
    input = Input(shape=[16, 66, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = Activation("relu", name='relu1')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = Activation("relu", name='relu2')(x)
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = Activation("relu", name='relu3')(x)
    x = Flatten()(x)
    output = Dense(2, name="dense")(x)
    output = Activation("relu", name='relu4')(output)
    model = Model([input], [output])
    return model


modelFineMapping = model_finemapping()
modelFineMapping.load_weights('./model/model12.h5')


def finemappingVertical(image, rect):
    resized = cv2.resize(image, (66, 16))
    resized = resized.astype(np.float) / 255
    res_raw = modelFineMapping.predict(np.array([resized]))[0]
    res = res_raw * image.shape[1]
    res = res.astype(np.int)
    H, T = res
    H -= 3
    if H < 0:
        H = 0
    T += 2
    if T >= image.shape[1] - 1:
        T = image.shape[1] - 1
    rect[2] -= rect[2] * (1 - res_raw[1] + res_raw[0])
    rect[0] += res[0]
    cv2.imshow("ww", image)
    image = image[:, H:T + 2]
    cv2.imshow("wd", image)
    image = cv2.resize(image, (int(136), int(36)))

    return image, rect


fontC = ImageFont.truetype("./model/platech.ttf", 14, 0)
importlib.reload(sys)


def drawRectBox(image, rect, addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 0, 255), 2,
                  cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0] + 1), int(rect[1] - 16)), addText.encode("utf-8").decode("utf-8"), (255, 255, 255),
              font=fontC)
    imagex = np.array(img)
    return imagex
