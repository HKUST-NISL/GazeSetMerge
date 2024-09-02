import cv2
import numpy as np
import torch
import torchvision

from torchvision import transforms

from PIL import Image

def flip_rect(rect, image_width=224):
    x1, y1, x2, y2 = rect

    y1_flip = y1
    y2_flip = y2
    x1_flip = image_width - x2
    x2_flip = image_width - x1

    rect[0] = x1_flip
    rect[1] = y1_flip
    rect[2] = x2_flip
    rect[3] = y2_flip

    return rect

def randdom_shift(rect, ratio=0.1, image_width=224):
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1

    wm = width * ratio
    hm = height * ratio

    x1 = int(x1 + (np.random.random() - 0.5) * wm)
    x2 = int(x2 + (np.random.random() - 0.5) * wm)
    y1 = int(y1 + (np.random.random() - 0.5) * hm)
    y2 = int(y2 + (np.random.random() - 0.5) * hm)

    x1 = max(0, x1)
    x2 = min(image_width-1, x2)
    y1 = max(0, y1)
    y2 = max(image_width-1, y2)

    rect[0] = x1
    rect[1] = y1
    rect[2] = x2
    rect[3] = y2

    return rect


class RandomMirror(object):
    def __init__(self, input_size):
        self.size = input_size[0]

    def __call__(self, x):

        # x['left_eye_box'] = randdom_shift(x['left_eye_box'], ratio=0.1, image_width=self.size)
        # x['right_eye_box'] = randdom_shift(x['right_eye_box'], ratio=0.1, image_width=self.size)

        flag = np.random.random()
        if flag > 0.5:
            x['mirror'] = 1
            return x

        image = x['image']
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        x['image'] = image

        x['mirror'] = -1
        
        if 'gaze' in x.keys():
            gaze = x['gaze']
            gaze[1] = -1 * gaze[1]
            x['gaze'] = gaze

        if 'gaze3d' in x.keys():
            gaze3d = x['gaze3d']
            gaze3d[1] = -1 * gaze3d[1]
            x['gaze3d'] = gaze3d
        
        if 'pose' in x.keys():
            pose = x['pose']
            pose[1] = -1 * pose[1]
            x['pose'] = pose

        if 'left_eye_box' in x.keys() and 'right_eye_box' in x.keys():
            left_eye_box = x['left_eye_box']
            right_eye_box = x['right_eye_box']
            right_eye_box, left_eye_box = flip_rect(left_eye_box, self.size), flip_rect(right_eye_box, self.size)
            x['left_eye_box'] = left_eye_box
            x['right_eye_box'] = right_eye_box

        return x


class ShowImage(object):

    def __call__(self, x):

        face = x['image']

        cv2.imshow('face', face)

        if 'left_eye_box' in x.keys() and 'right_eye_box' in x.keys():
            left_eye_box = x['left_eye_box']
            right_eye_box = x['right_eye_box']

            left_eye = self.crop_eye(face, left_eye_box)
            right_eye = self.crop_eye(face, right_eye_box)

            cv2.imshow('left_eye', left_eye)
            cv2.imshow('right_eye', right_eye)

        cv2.waitKey(0)
            
        return x

    def crop_eye(self, face, rect):
        x1, y1, x2, y2 = [int(x) for x in rect]
        eye = face[y1:y2, x1:x2]
        return eye

class NumpyToPIL(object):

    def __init__(self):
        self.topil = transforms.ToPILImage()

    def __call__(self, x):

        image = x['image']
        image = image[..., ::-1]   # from BGR to RGB
        image = self.topil(image)
        x['image'] = image

        return x


class TensorNormize(object):
    def __init__(self):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        # means = [0.5, 0.5, 0.5]
        # stds = [0.5, 0.5, 0.5]
        self.norm_fuc = torchvision.transforms.Normalize(mean=means,std=stds)
    
    def __call__(self, x):

        image = x['image']
        image = self.norm_fuc(image)
        x['image'] = image

        return x

class ImagetoTensor(object):

    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, x):

        x['image'] = self.totensor(x['image'])

        return x


class ImageResize(object):

    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, x):
        tw, th = self.input_size

        image = x['image']
        # print('before:', x['left_eye_box'])
        w, h = image.size
        if tw != w or th != h:
            image = image.resize((tw, th))
            x['image'] = image
            left, top, right, bottom = x['left_eye_box']
            x['left_eye_box'] = np.array([left/w*tw, top/h*th, right/w*tw, bottom/w*tw])
            left, top, right, bottom = x['right_eye_box']
            x['right_eye_box'] = np.array([left/w*tw, top/h*th, right/w*tw, bottom/w*tw])
            # print('after:', x['left_eye_box'])

        return x


def create_transform(input_size, is_train):

    show_images = ShowImage()
    transform_topil = NumpyToPIL()
    transform_totensor = ImagetoTensor()
    transform_norm = TensorNormize()
    transform_resize = ImageResize(input_size)
    
    # trans for train
    transform_mirror = RandomMirror(input_size)

    if not is_train:
        transform = torchvision.transforms.Compose([
            # show_images,
            transform_topil,
            transform_resize,
            transform_totensor,
            transform_norm,
            ])
        return transform

    transform = torchvision.transforms.Compose([
        # show_images,
        transform_topil,
        transform_resize,
        transform_mirror,
        transform_totensor,
        transform_norm,
        ])
    return transform