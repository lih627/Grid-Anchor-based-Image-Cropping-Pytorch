from croppingModel import build_crop_model
from croppingDataset import setup_test_dataset, TransformFunctionTest
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.autograd import Variable
import cv2
import random
import os
import numpy as np
from utils import resize_image_op, color_normalization_op, transpose_image_op, generate_bboxs


def generate_dataset(path):
    return setup_test_dataset(path)


class AutoCrop(object):

    def __init__(self,
                 cuda=True,
                 net_path='pretrained_model/mobilenet_0.625_0.583'
                          '_0.553_0.525_0.785_0.762_0.748_0.723_0.'
                          '783_0.806.pth',
                 ):
        self.net = build_crop_model(scale='multi',
                                    alignsize=9,
                                    reddim=8,
                                    loadweight=False,
                                    model='mobilenetv2',
                                    downsample=4)
        self.net.load_state_dict(torch.load(net_path))
        self.net.eval()
        self.cuda = cuda
        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            self.net.cuda()

    def autoCropPlain(self,
                      img_path=None,
                      topK=2,
                      show_ret=False,
                      save_ret=True,
                      save_dir='./dataset',
                      image_size=256.0,
                      debug=False):
        """
        Copy from raw GAIC codebase, modified by lihao 2020

        :param img_path: str, image file path
        :param topK: int, show topK cropped results
        :param show_ret: bool, show the crop result by cv2 if True
        :param save_ret: bool, save the result if True
        :param save_dir:  str, the directory that cropped result saved in
        :param image_size: float, default 256 for the pretrained model
        :param debug: bool, show debug information if Ture
        :return: None
        """

        assert img_path is not None, 'Please input image path'
        if not isinstance(image_size, float):
            image_size = float(image_size)
        bgr_img = cv2.imread(img_path)
        rgb_img = bgr_img[:, :, (2, 1, 0)]
        transform_func = TransformFunctionTest()

        (resized_rgb_img, transformed_bbox, source_bboxes,
         raw_resized_rgb_img, trans_bboxes) = transform_func(rgb_img, image_size)

        # sample = {'imgpath': img_path, 'image': rgb_img, 'resized_image': resized_rgb_img,
        #           'tbboxes': transformed_bbox, 'sourceboxes': source_bboxes}

        roi = []

        for idx, tbbox in enumerate(trans_bboxes):
            roi.append((0, *tbbox))
        resized_rgb_img_torch = torch.unsqueeze(torch.as_tensor(resized_rgb_img), 0)
        if self.cuda:
            resized_rgb_img_torch = Variable(resized_rgb_img_torch.cuda())
            roi = Variable(torch.Tensor(roi))
        else:
            resized_rgb_img_torch = Variable(resized_rgb_img_torch)
            roi = Variable(torch.Tensor(roi))
        out = self.net(resized_rgb_img_torch, roi)
        out = out
        id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)
        selected_ids = id_out[:topK]
        top_crops = []
        top_bboxs = []
        for id_ in selected_ids:
            cbbox = source_bboxes[id_]
            top_bboxs.append(cbbox)
            top_crops.append(rgb_img[int(cbbox[0]): int(cbbox[2]), int(cbbox[1]): int(cbbox[3]), :])

        if show_ret:
            cv2.imshow('original image', bgr_img)
            for nth, c_crop in enumerate(top_crops):
                cv2.imshow('Top{} Crop'.format(nth + 1), c_crop[:, :, (2, 1, 0)])

        if debug:
            # raw_resized_bgr_img = cv2.UMat(raw_resized_rgb_img[:, :, (2, 1, 0)])
            bgr_img_with_anchor = bgr_img.copy()
            for idx, bbox_source in enumerate(source_bboxes):
                font = cv2.FONT_HERSHEY_TRIPLEX
                dx1, dx2, dy1, dy2 = int((random.random() - 1) * 4), int((random.random() - 1) * 4), \
                                     int((random.random() - 1) * 4), int((random.random() - 1) * 4)
                r, g, b = int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
                cv2.rectangle(bgr_img_with_anchor,
                              tuple([bbox_source[1] + dx1, bbox_source[0] + dy1]),
                              tuple([bbox_source[3] + dx2, bbox_source[2] + dy2]),
                              (b, g, r))
                cv2.putText(bgr_img_with_anchor, str(idx + 1),
                            tuple([bbox_source[1] + dx1, bbox_source[0] + dy1]),
                            font, 0.5, (0, 0, 0))
            cv2.imshow('original img with anchor-num anchor:{}'.format(len(source_bboxes)),
                       bgr_img_with_anchor)
            bgr_img_with_top_k = bgr_img.copy()
            assert bgr_img_with_anchor is not bgr_img_with_top_k, 'DEBUG show ret'
            for idx, tbox in enumerate(top_bboxs):
                font = cv2.FONT_HERSHEY_TRIPLEX
                dx1, dx2, dy1, dy2 = int((random.random() - 1) * 4), int((random.random() - 1) * 4), \
                                     int((random.random() - 1) * 4), int((random.random() - 1) * 4)
                r, g, b = int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
                cv2.rectangle(bgr_img_with_top_k,
                              tuple([tbox[1] + dx1, tbox[0] + dy1]),
                              tuple([tbox[3] + dx2, tbox[2] + dy2]),
                              (b, g, r))
                cv2.putText(bgr_img_with_top_k, 'TOP{}'.format(idx + 1),
                            tuple([tbox[1] + dx1, tbox[0] + dy1]),
                            font, 0.5, (0, 0, 0))
            cv2.imshow('Original detection TOP{}'.format(len(top_bboxs)), bgr_img_with_top_k)

        if debug or show_ret:
            cv2.waitKey()

        if save_ret:
            img_name = os.path.basename(img_path)
            segs = img_name.split('.')
            assert len(segs) >= 2
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for idx, corp_ret in enumerate(top_crops):
                save_path = os.path.join(save_dir,
                                         '.'.join(segs[:-1] + '_{}'.format(idx + 1) + '.' + segs[-1]))

                cv2.imwrite(save_path, corp_ret[:, :, (2, 1, 0)])

    def network_eval(self, image, roi):
        trans_image = transpose_image_op(image)
        image = torch.unsqueeze(torch.as_tensor(trans_image), 0)
        if self.cuda:
            resized_rgb_img_torch = Variable(image.cuda())
            roi = Variable(torch.Tensor(roi))
        else:
            resized_rgb_img_torch = Variable(image)
            roi = Variable(torch.Tensor(roi))
        out = self.net(resized_rgb_img_torch, roi)
        id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)
        return id_out

    def autoCrop(self,
                 img_path=None,
                 topK=2,
                 crop_height=None,
                 crop_width=None,
                 show_ret=False,
                 save_ret=True,
                 save_dir='./dataset',
                 image_size=256.0,
                 debug=False):

        assert img_path is not None, 'Please input image path'
        assert os.path.exists(img_path), 'Can not find {}'.format(img_path)
        if not isinstance(image_size, float):
            image_size = float(image_size)
        bgr_img = cv2.imread(img_path)
        rgb_img = bgr_img[:, :, (2, 1, 0)]
        resized_rgb_img, scl_h, scl_w = resize_image_op(rgb_img, image_size)
        resized_rgb_img_norm = color_normalization_op(resized_rgb_img)
        assert resized_rgb_img is not resized_rgb_img_norm, 'DEBUG'
        trans_bboxes, source_bboxes = generate_bboxs(resized_rgb_img,
                                                     scale_height=scl_h,
                                                     scale_width=scl_w,
                                                     crop_height=crop_height,
                                                     crop_width=crop_width)
        roi = []
        for idx, tbbox in enumerate(trans_bboxes):
            roi.append((0, *tbbox))
        id_out = self.network_eval(resized_rgb_img_norm, roi)
        selected_ids = id_out[:topK]
        top_crops = []
        top_bboxs = []
        for id_ in selected_ids:
            # cbbox ymin ximn ymax xmax
            cbbox = source_bboxes[id_]
            top_bboxs.append(cbbox)
            # H W C ymin:ymax xmin: xmax
            top_crops.append(rgb_img[int(cbbox[0]): int(cbbox[2]), int(cbbox[1]): int(cbbox[3]), :])

        if show_ret:
            cv2.imshow('original image', bgr_img)
            for nth, c_crop in enumerate(top_crops):
                cv2.imshow('Top{} Crop'.format(nth + 1), c_crop[:, :, (2, 1, 0)])

        if debug:
            bgr_img_with_anchor = bgr_img.copy()
            for idx, bbox_source in enumerate(source_bboxes):
                font = cv2.FONT_HERSHEY_TRIPLEX
                dx1, dx2, dy1, dy2 = int((random.random() - 1) * 4), int((random.random() - 1) * 4), \
                                     int((random.random() - 1) * 4), int((random.random() - 1) * 4)
                r, g, b = int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
                cv2.rectangle(bgr_img_with_anchor,
                              # xmin ymin xmax ymax
                              #    xmin, ymin
                              #   |
                              #   |
                              #   |___________xmax, ymax
                              tuple([bbox_source[1] + dx1, bbox_source[0] + dy1]),
                              tuple([bbox_source[3] + dx2, bbox_source[2] + dy2]),
                              (b, g, r))
                cv2.putText(bgr_img_with_anchor, str(idx + 1),
                            tuple([bbox_source[1] + dx1, bbox_source[0] + dy1]),
                            font, 0.5, (0, 0, 0))
            cv2.imshow('original img with anchor-num anchor:{}'.format(len(source_bboxes)),
                       bgr_img_with_anchor)
            bgr_img_with_top_k = bgr_img.copy()
            # print(bgr_img_with_anchor is bgr_img_with_top_k)
            for idx, tbox in enumerate(top_bboxs):
                font = cv2.FONT_HERSHEY_TRIPLEX
                dx1, dx2, dy1, dy2 = int((random.random() - 1) * 4), int((random.random() - 1) * 4), \
                                     int((random.random() - 1) * 4), int((random.random() - 1) * 4)
                r, g, b = int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
                cv2.rectangle(bgr_img_with_top_k,
                              tuple([tbox[1] + dx1, tbox[0] + dy1]),
                              tuple([tbox[3] + dx2, tbox[2] + dy2]),
                              (b, g, r))
                cv2.putText(bgr_img_with_top_k, 'TOP{}'.format(idx + 1),
                            tuple([tbox[1] + dx1, tbox[0] + dy1]),
                            font, 0.5, (0, 0, 0))
            cv2.imshow('Original detection TOP{}'.format(len(top_bboxs)), bgr_img_with_top_k)

        if debug or show_ret:
            cv2.waitKey()

        if save_ret:
            img_name = os.path.basename(img_path)
            segs = img_name.split('.')
            assert len(segs) >= 2
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for idx, corp_ret in enumerate(top_crops):
                save_path = os.path.join(save_dir,
                                         '.'.join(segs[:-1]) +
                                         '_{}'.format(idx + 1) +
                                         '.' + segs[-1])

                cv2.imwrite(save_path, corp_ret[:, :, (2, 1, 0)])

    def autoCropHuman(self):
        raise NotImplementedError

    def autoCropText(self):
        raise NotImplementedError


if __name__ == '__main__':
    autoCrop = AutoCrop()
    for w, h in [(210, 294),
                    (342, 478),
                    (525, 295),
                    (716, 229),
                    (1125, 1125)]:
        save_dir = './dataset/ret/{}_{}'.format(w, h)
        for item  in os.listdir('./dataset/eval'):
            img_path = './dataset/eval/' + item
            autoCrop.autoCrop(img_path=img_path,
                              topK=3,
                              crop_width=w,
                              crop_height=h,
                              show_ret=False,
                              save_ret=True,
                              save_dir=save_dir,
                              debug=False)


    # autoCrop.autoCrop(img_path='./dataset/test_4.jpg',
    #                   topK=3,
    #                   crop_height=10,
    #                   crop_width=16,
    #                   show_ret=True,
    #                   save_ret=True,
    #                   save_dir='./result',
    #                   debug=True)
