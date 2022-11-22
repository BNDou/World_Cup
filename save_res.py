# -*- coding:utf-8 -*-

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

with open("country_name.txt", "r", encoding="utf-8") as f:
    info = f.readlines()
    info = list(map(lambda x:x.strip(), info))

English_name = info[:32]
Chinese_name = info[32:]

country_name = {}

for each in zip(English_name, Chinese_name):
    country_name[each[0]] = each[1]

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def save_res(a, b, winner, prob, save_name):
    img = np.zeros((520, 850, 3), np.uint8)

    a_img_path = "./world_cup/" + country_name[a] + ".png"
    a_img = cv2.imdecode(np.fromfile(a_img_path, dtype = np.uint8), -1)

    a_img_h, a_img_w, _ = a_img.shape

    b_img_path = "./world_cup/" + country_name[b] + ".png"
    b_img = cv2.imdecode(np.fromfile(b_img_path, dtype = np.uint8), -1)

    b_img_h, b_img_w, _ = b_img.shape

    winner_img_path = "./world_cup/" + country_name[winner] + ".png"
    winner_img = cv2.imdecode(np.fromfile(winner_img_path, dtype = np.uint8), -1)

    winner_img_h, winner_img_w, _ = winner_img.shape

    img[10: 10 + a_img_h, 10: 10 + a_img_w] = a_img[:, :, :3]

    img[110 + a_img_h: 110 + a_img_h + b_img_h, 10: 10 + b_img_w] = b_img[:,:,:3]


    img = cv2AddChineseText(img, country_name[a] + "(" + a + ")", (10, 20 + a_img_h),(255, 255, 255), 30)

    img = cv2AddChineseText(img, country_name[b] + "(" + b + ")", (10, 20 + 100 + a_img_h + b_img_h), (255, 255, 255), 30)

    point1 = (10 + a_img_w, 10 + (a_img_h) // 2)
    point2 = (10 + a_img_w + 100, 10 + (a_img_h) // 2)

    cv2.line(img, point1, point2, (255, 255, 255), 10)

    point3 = (10 + a_img_w, 10 + 100 + a_img_h + (b_img_h) // 2)
    point4 = (10 + b_img_w + 100, 10 + 100 + a_img_h + (b_img_h) // 2)

    cv2.line(img, point3, point4, (255, 255, 255), 10)

    cv2.line(img, point2, point4, (255, 255, 255), 10)


    point5 = (10 + a_img_w + 100, 10 + 50 + a_img_h)
    point6 = (10 + a_img_w + 300, 10 + 50 + a_img_h)
    cv2.line(img, point5, point6, (255, 255, 255), 10)

    img = cv2AddChineseText(img, "胜率：{}".format(prob), (10 + a_img_w + 100 + 20, 10 + 50 + a_img_h - 40),(255, 255, 255), 30)

    img[10 + 50 + a_img_h - winner_img_h // 2: 10 + 50 + a_img_h + winner_img_h // 2, 10 + a_img_w + 300 : 10 + a_img_w + 300 + winner_img_w] = winner_img[:,:,:3]

    img = cv2AddChineseText(img, country_name[winner] + "(" + winner + ")", (10 + a_img_w + 300, 10 + 50 + a_img_h + winner_img_h // 2 + 10), (255, 255, 255), 30)

    cv2.imwrite(save_name, img)

def save_res_draw(a, b, winner, prob, save_name):
    img = np.zeros((520, 850, 3), np.uint8)

    a_img_path = "./world_cup/" + country_name[a] + ".png"
    a_img = cv2.imdecode(np.fromfile(a_img_path, dtype = np.uint8), -1)

    a_img_h, a_img_w, _ = a_img.shape

    b_img_path = "./world_cup/" + country_name[b] + ".png"
    b_img = cv2.imdecode(np.fromfile(b_img_path, dtype = np.uint8), -1)

    b_img_h, b_img_w, _ = b_img.shape

    winner_img_path = "./world_cup/" + country_name[winner] + ".png"
    winner_img = cv2.imdecode(np.fromfile(winner_img_path, dtype = np.uint8), -1)

    winner_img_h, winner_img_w, _ = winner_img.shape

    img[10: 10 + a_img_h, 10: 10 + a_img_w] = a_img[:, :, :3]

    img[110 + a_img_h: 110 + a_img_h + b_img_h, 10: 10 + b_img_w] = b_img[:,:,:3]


    img = cv2AddChineseText(img, country_name[a] + "(" + a + ")", (10, 20 + a_img_h),(255, 255, 255), 30)

    img = cv2AddChineseText(img, country_name[b] + "(" + b + ")", (10, 20 + 100 + a_img_h + b_img_h), (255, 255, 255), 30)

    point1 = (10 + a_img_w, 10 + (a_img_h) // 2)
    point2 = (10 + a_img_w + 100, 10 + (a_img_h) // 2)

    cv2.line(img, point1, point2, (255, 255, 255), 10)

    point3 = (10 + a_img_w, 10 + 100 + a_img_h + (b_img_h) // 2)
    point4 = (10 + b_img_w + 100, 10 + 100 + a_img_h + (b_img_h) // 2)

    cv2.line(img, point3, point4, (255, 255, 255), 10)

    cv2.line(img, point2, point4, (255, 255, 255), 10)


    point5 = (10 + a_img_w + 100, 10 + 50 + a_img_h)
    point6 = (10 + a_img_w + 300, 10 + 50 + a_img_h)
    cv2.line(img, point5, point6, (255, 255, 255), 10)

    img = cv2AddChineseText(img, "胜率：{}".format(prob), (10 + a_img_w + 100 + 20, 10 + 50 + a_img_h - 40),(255, 255, 255), 30)

    # img[10 + 50 + a_img_h - winner_img_h // 2: 10 + 50 + a_img_h + winner_img_h // 2, 10 + a_img_w + 300 : 10 + a_img_w + 300 + winner_img_w] = winner_img[:,:,:3]

    # img = cv2AddChineseText(img, country_name[winner] + "(" + winner + ")", (10 + a_img_w + 300, 10 + 50 + a_img_h + winner_img_h // 2 + 10), (255, 255, 255), 30)

    cv2.imwrite(save_name, img)

if __name__ == "__main__":
    save_res_draw("Switzerland", "Cameroon", "Switzerland", 0.62, "tmp.png")