import cv2
from pytesseract import Output
import pytesseract
from PIL import Image
"""
对于艺术字检测效果很差
不能使用
"""
img = cv2.imread('./dataset/word_2.jpg')
d = pytesseract.image_to_data(img, output_type=Output.DICT, lang='chi_sim')
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
print(d)
cv2.imshow('img', img)
cv2.waitKey(0)