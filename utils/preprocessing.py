import cv2
import numpy as np


def load_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")

    return img


def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_gaussian_filter(gray_img, kernel_size=(5, 5)):
    return cv2.GaussianBlur(gray_img, kernel_size, 0)


def apply_median_filter(gray_img, kernel_size=5):
    return cv2.medianBlur(gray_img, kernel_size)


def enhance_contrast(gray_img):
    return cv2.equalizeHist(gray_img)


def preprocess_image(image_path, filter_type="gaussian", enhance=True):
    img = load_image(image_path)
    gray = convert_to_grayscale(img)

    if filter_type == "gaussian":
        filtered = apply_gaussian_filter(gray)
    elif filter_type == "median":
        filtered = apply_median_filter(gray)
    else:
        filtered = gray

    if enhance:
        final_img = enhance_contrast(filtered)
    else:
        final_img = filtered

    return final_img