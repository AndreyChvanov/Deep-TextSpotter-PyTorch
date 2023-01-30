import numpy as np
import random
from PIL import Image


class VerticalResize:
    def __init__(self, new_h):
        self.h = new_h

    def __call__(self, image):
        w, h = image.size
        w_ = self.h * w // h
        return image.resize((w_, self.h))


class HorizontalResize:
    def __init__(self, max_w):
        self.max_w = max_w

    def __call__(self, image):
        w, h = image.size
        if w > self.max_w:
            return image.resize((self.max_w, h))
        return image


class RandomHorizontalResize:
    def __init__(self, scale=(0.8, 1.2), prob=1):
        self.scale = scale
        self.possible_scale = np.linspace(self.scale[0], self.scale[1], 50)
        self.prob = prob

    def __call__(self, image):
        p = np.random.randint(0, 10)
        if not (p < self.prob * 10):
            return image
        w, h = image.size
        cur_scale = self.possible_scale[random.randint(0, len(self.possible_scale))]
        new_w = int(cur_scale * w)
        return image.resize((new_w, h))


class RandomHorizontalCrop:
    def __init__(self, scale=(0.01, 0.05), prob=1):
        self.scale = scale
        self.possible_scales = np.linspace(self.scale[0], self.scale[1], 10)
        self.prob = prob

    def __call__(self, image):
        p = np.random.randint(0, 10)
        if not (p < self.prob * 10):
            return image
        w, h = image.size
        left_scale = self.possible_scales[random.randint(0, len(self.possible_scales))]
        right_scale = self.possible_scales[random.randint(0, len(self.possible_scales))]
        w_left, right_w = left_scale * w, w - w * right_scale
        return image.crop((w_left, 0, right_w, h))


def gaussian_noise(image_shape, mean, var):
    noise = np.random.normal(mean, var ** 0.5, image_shape).reshape(image_shape).astype(np.uint8)

    return noise


class RandomGaussianNoise:
    def __init__(self, mean, std, prob):
        self.mean, self.std, self.prob = mean, std, prob

    def __call__(self, image):
        p = np.random.randint(0, 10)
        if not (p < self.prob * 10):
            return image
        w, h = image.size
        noise = gaussian_noise((h, w, 3), self.mean, self.std)
        return Image.fromarray((image + noise).astype(np.uint8))
