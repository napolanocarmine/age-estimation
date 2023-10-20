#from TFRecordBilanciato import *
import numpy as np
import cv2
import random
from Tools import*


def contrast(image):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    contrast_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

    return contrast_image


def white_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1)
    white_image = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return white_image


def random_gaussian_noise(image):
    if random.randint(0, 1):
        row, col, ch = image.shape
        binom = np.random.binomial(n=500, p=0.5, size=(row, col, ch))
        binom = binom - 250
        noisy = image + binom
        noisy = np.clip(noisy, 0, 255)
        image = noisy.astype(np.uint8)
    return image


def random_brightness(image):
    if random.randint(0, 1):
        esposizione = random.randint(1, 2)  # se 1 sovraespongo se 2 sottoespongo
        if esposizione == 2:
            brigh = image * 0.7
        else:
            brigh = image * 1.4
        brigh = np.clip(brigh, 0, 255)
        image = brigh.astype(np.uint8)
    return image


def random_flip(image):
    if random.randint(0, 1):
        flip = np.fliplr(image)
        image = flip
        image = flip.astype(np.uint8)
    return image


def hue_image(image, saturation=50):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def erosion_image(image, shift=3):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image


def bileteralBlur(image, d=40, color=75, space=75):
    image = cv2.bilateralFilter(image, d, color, space)
    return image


def salt_and_paper_image(image, p=0.6, a=0.09):
    noisy = image
    # salt
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1
    # paper
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    return image


def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image


def top_hat_image(image, shift=200):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return image


def shear_image(img, shear_range=10):
    rows, cols, ch = img.shape
    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_M = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, shear_M, (cols, rows))
    return img



def basic_augmentation(original, writer, filename, label):
    """every image is saved 2 times with 1 filter"""
    # filtro 1 = flip
    image_flip = random_flip(original)
    save_and_write(writer, filename + '_flip', label, image_flip)
    return image_flip


def medium_augmentation(original, writer, filename, label):
    """every image is saved 3 times with 2 different filters"""
    basic = basic_augmentation(original, writer, filename, label)
    # filtro 2 = gaussian_noise
    image_gauss = random_gaussian_noise(original)
    save_and_write(writer, filename + '_gauss', label, image_gauss)
    return basic, image_gauss


def high_augmentation(original, writer, filename, label):
    """every image is saved 5 times with 4 different filters"""
    basic, medium = medium_augmentation(original, writer, filename, label)
    # filtro 3 = flip, brightness
    image_flip_brightness = random_brightness(basic)
    save_and_write(writer, filename + '_flip_bright', label, image_flip_brightness)
    # filtro 4 = flip, gaussian noise
    image_flip_gauss = random_gaussian_noise(basic)
    save_and_write(writer, filename + '_flip_gauss', label, image_flip_gauss)
    return basic, medium, image_flip_brightness, image_flip_gauss


def extreme_augmentation(original, writer, filename, label):
    "every image is saved 10 times with 9 different filters"
    basic, medium, high1, high2 = high_augmentation(original, writer, filename, label)
    # filtro 5 = gaussian noise, hue
    image_gauss_hue = hue_image(medium)
    save_and_write(writer, filename + '_gauss_hue', label, image_gauss_hue)
    # filtro 6 = flip,gaussian,brightnes
    image_flip_gauss_brightness = random_brightness(high2)
    save_and_write(writer, filename + '_flip_gauss_bright', label, image_flip_gauss_brightness)
    # filtro 7 = erosion
    image_erosion = erosion_image(original)
    save_and_write(writer, filename + '_erosion', label, image_erosion)
    # filtro 8 = shear
    image_shear = shear_image(original)
    save_and_write(writer, filename + '_shear', label, image_shear)
    # flitro 9 = salt and pepper
    image_salt_and_pepper = salt_and_paper_image(original)
    save_and_write(writer, filename + '_sp', label, image_salt_and_pepper)

