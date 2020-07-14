import cv2
import numpy as np


def resize_image_op(raw_img, img_size=256.0):
    """
    :param raw_img: [H, W, 3]
    :param img_size: float: default 256.0 for pretrained model
    :return:
        resized_img: [H', 256, 3] or [256, W', 3]
        scale_height: float H/resized_H
        scale_width: float W/resized_W
    """
    scale = float(img_size) / float(min(raw_img.shape[:2]))
    h = round(raw_img.shape[0] * scale / 32.0) * 32
    w = round(raw_img.shape[1] * scale / 32.0) * 32

    resized_img = cv2.resize(raw_img, (int(w), int(h)))
    scale_height = raw_img.shape[0] / float(resized_img.shape[0])
    scale_width = raw_img.shape[1] / float(resized_img.shape[0])
    return resized_img, scale_height, scale_width


def color_normalization_op(image):
    """
    :param image: [H, W, 3] RGB format
    :return: image: [H, W, 3] color normalization
    """
    RGB_MEAN = (0.485, 0.456, 0.406)
    RGB_STD = (0.229, 0.224, 0.225)
    image = image.astype(np.float32)
    image /= 256.0
    rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
    rgb_std = np.array(RGB_STD, dtype=np.float32)
    image -= rgb_mean
    image /= rgb_std
    return image


def transpose_image_op(image):
    """
    Transpose image as [C, H , W]
    :param image:  [H, W, 3]
    :return:
        [3, H,  W]
    """
    image = image.transpose((2, 0, 1))
    return image


def _generate_by_bins(image, n_bins=12):
    """
    :param image:
    :param n_bins:
    :return:
       ---------> w
       |
       |
       |
       | h
       bboxs:  N * 4 [h1, w1, h2, w2]
    """
    h = image.shape[0]
    w = image.shape[1]
    step_h = h / float(n_bins)
    step_w = w / float(n_bins)
    annotations = list()
    front = n_bins // 3 + 1
    back = n_bins // 3 * 2

    for x1 in range(0, front):
        for y1 in range(0, front):
            for x2 in range(back, n_bins):
                for y2 in range(back, n_bins):
                    if (x2 - x1) * (y2 - y1) > 0.4 * n_bins * n_bins and (y2 - y1) * step_w / (
                            x2 - x1) / step_h > 0.5 and (y2 - y1) * step_w / (x2 - x1) / step_h < 2.0:
                        annotations.append(
                            [float(step_h * (0.5 + x1)), float(step_w * (0.5 + y1)), float(step_h * (0.5 + x2)),
                             float(step_w * (0.5 + y2))])
    return annotations


def _generate_by_steps(image, height_step, width_step):
    h, w = image.shape[:2]
    h_bins = h // height_step
    w_bins = w // width_step
    annotations = []
    for h_bin_end in range(h_bins // 3, h_bins + 1):
        for w_bin_end in range(w_bins // 3, w_bins + 1):
            h_end = h_bin_end * height_step
            w_end = w_bin_end * width_step
            if h_end < h and w_end < w:
                for length in range(min(h_bin_end, w_bin_end), 1, -1):
                    h_len = length * height_step
                    w_len = length * width_step
                    h_start = h_end - h_len
                    w_start = w_end - w_len
                    if w_len * h_len > 0.2 * w * h:
                        annotations.append([h_start, w_start, h_end, w_end])
    return annotations


def generate_bboxs(resized_image,
                   scale_height,
                   scale_width,
                   crop_height=None,
                   crop_width=None,
                   min_step=8,
                   n_bins=14):
    """
    generate tansformed_bbox and source_bbox
    :param resized_image:
    :param scale_height:
    :param scale_width:
    :param crop_height:
    :param crop_width:
    :param min_step: the minimum step_size: defalut 8
    :return:
        trans_bboxs  [Nx4] [y1 x1 y2 x2]
        source_bboxs [Nx4] [y1 x1 y2 x2]
    """
    if crop_height is None or crop_width is None:
        bboxs = _generate_by_bins(resized_image, n_bins=n_bins)
    else:
        h_w_ratio = crop_height / float(crop_width)
        if h_w_ratio < 1.0:
            h_step = min_step
            w_step = round(crop_width * min_step / float(crop_height))
        else:
            w_step = min_step
            h_step = round(crop_height * min_step) / float(crop_width)
        bboxs = _generate_by_steps(resized_image, height_step=h_step, width_step=w_step)
    source_bboxs = []
    trans_bboxs = []
    for bbox in bboxs:
        # h1, w1, h2: w2 [y1 x1 y2 x2]  [1, 0, 3, 2]
        trans_bboxs.append([round(item) for item in [bbox[1], bbox[0],
                                                     bbox[3], bbox[2]]])
        source_bboxs.append([round(item) for item in [bbox[1] * scale_height,
                                                      bbox[0] * scale_width,
                                                      bbox[3] * scale_height,
                                                      bbox[2] * scale_width]])
    return trans_bboxs, source_bboxs
