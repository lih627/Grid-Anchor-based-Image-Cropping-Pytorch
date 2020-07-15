import cv2
from pytesseract import Output
import pytesseract

import face_detection


def testTextDetection():
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


def enlarge(bbox, factor=1.2):
    """
    Enlarge the bounding boxes
    :param bbox: [N * C] array
    :return:
        [N * 4] xmin ymin xmax ymax
    """
    xmin, ymin, xmax, ymax = bbox[:4]
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w_ = (xmax - xmin) * factor / 2.0
    h_ = (ymax - ymin) * factor / 2.0
    return  [cx - w_, cy - h_, cx + w_, cy + h_]


print(face_detection.available_detectors)

img = cv2.imread('./dataset/test_1.jpg')
img_ = img[:, :, (2, 1, 0)]
detector = face_detection.build_detector(
    "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

detections = detector.detect(img_)

print(detections)
for bbox in detections:
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0))
    c = enlarge(bbox)
    print(c)
    cv2.rectangle(img, (c[0], c[1]), (c[2], c[3]), (255, 0, 0))
cv2.imshow('ret', img)
cv2.waitKey()
