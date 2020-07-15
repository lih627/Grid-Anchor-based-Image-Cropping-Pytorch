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
from utils import resize_image_op, color_normalization_op, transpose_image_op, generate_bboxes, enlarge_bbox
import face_detection


def generate_dataset(path):
    return setup_test_dataset(path)


class AutoCrop(object):

    def __init__(self,
                 cuda=True,
                 net_path='pretrained_model/mobilenet_0.625_0.583'
                          '_0.553_0.525_0.785_0.762_0.748_0.723_0.'
                          '783_0.806.pth',
                 add_face_detector=True
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
        self.face_detector = None
        if add_face_detector:
            self.face_detector = face_detection.build_detector("DSFDDetector",
                                                                confidence_threshold=0.5,
                                                                nms_iou_threshold=0.3)
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

        roi = []

        for idx, tbbox in enumerate(trans_bboxes):
            roi.append((0, *tbbox))
        id_out = self.network_eval(resized_rgb_img)
        selected_ids = id_out[:topK]
        top_crops = []
        top_bboxes = []
        for id_ in selected_ids:
            cbbox = source_bboxes[id_]
            top_bboxes.append(cbbox)
            top_crops.append(rgb_img[int(cbbox[0]): int(cbbox[2]), int(cbbox[1]): int(cbbox[3]), :])

        if show_ret:
            self.cv2_show_ret(bgr_img, top_crops)

        if debug:
            self._debug(bgr_img, source_bboxes, top_bboxes)

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
        face_bboxes = None
        if self.face_detector is not None:
            face_bboxes = self.face_detector.detect(resized_rgb_img)
            face_bboxes = [enlarge_bbox(bbox) for bbox in face_bboxes]


        trans_bboxes, source_bboxes = generate_bboxes(resized_rgb_img,
                                                      scale_height=scl_h,
                                                      scale_width=scl_w,
                                                      crop_height=crop_height,
                                                      crop_width=crop_width,
                                                      face_bboxes=face_bboxes)
        roi = []
        for idx, tbbox in enumerate(trans_bboxes):
            roi.append((0, *tbbox))
        id_out = self.network_eval(resized_rgb_img_norm, roi)
        selected_ids = id_out[:topK]
        top_crops = []
        top_bboxes = []
        for id_ in selected_ids:
            # cbbox ymin ximn ymax xmax
            cbbox = source_bboxes[id_]
            top_bboxes.append(cbbox)
            # H W C ymin:ymax xmin: xmax
            top_crops.append(rgb_img[int(cbbox[0]): int(cbbox[2]), int(cbbox[1]): int(cbbox[3]), :])

        if show_ret:
            cv2.imshow('original image', bgr_img)
            for nth, c_crop in enumerate(top_crops):
                cv2.imshow('Top{} Crop'.format(nth + 1), c_crop[:, :, (2, 1, 0)])

        if debug:
            self._debug(bgr_img, source_bboxes, top_bboxes)

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

    def _debug(self, bgr_img, source_bboxes, top_bboxes):
        """
        show bgr image with anchor bboxes and top_bboxs
        :param bgr_img:
        :param source_bboxes:
        :param top_bboxes:
        :return:
        """
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
        for idx, tbox in enumerate(top_bboxes):
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
        cv2.imshow('Original detection TOP{}'.format(len(top_bboxes)), bgr_img_with_top_k)

    def cv2_show_ret(self, org_img, top_crops):
        """
        :param org_img: [H W 3] BGR MODE
        :param top_crops: [H W 3] RGB MODE
        :return:
        """
        cv2.imshow('original image', org_img)
        for nth, c_crop in enumerate(top_crops):
            cv2.imshow('Top{} Crop'.format(nth + 1), c_crop[:, :, (2, 1, 0)])

    def network_eval(self, image, roi):
        """
        :param image:
        :param roi:
        :return:
            id_out
        """
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


if __name__ == '__main__':
    autoCrop = AutoCrop()
    for item in os.listdir('./dataset/multi-person'):
        img_path = './dataset/multi-person/' + item
        for w, h in [(210, 294),
                     (342, 478),
                     (525, 295),
                     (716, 229),
                     (1125, 1125)]:
            save_dir = './dataset/ret/multi_{}_{}'.format(w, h)

            autoCrop.autoCrop(img_path=img_path,
                              topK=3,
                              crop_width=w,
                              crop_height=h,
                              show_ret=False,
                              save_ret=True,
                              save_dir=save_dir,
                              debug=False)

    # autoCrop.autoCrop(img_path='./dataset/test_5.jpg',
    #                   topK=1,
    #                   crop_height=9,
    #                   crop_width=16,
    #                   show_ret=True,
    #                   save_ret=True,
    #                   save_dir='./result',
    #                   debug=True)
