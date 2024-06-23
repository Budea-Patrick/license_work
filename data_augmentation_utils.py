import cv2 as opencv
import numpy as np
import random

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = opencv.getRotationMatrix2D(center, angle, 1.0)
    rotated = opencv.warpAffine(image, M, (w, h))
    return rotated

def flip_image(image):
    return opencv.flip(image, 1)  # Horizontal flip

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.05
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss.reshape(row, col, ch)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def scale_image(image, scale):
    (h, w) = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = opencv.resize(image, (new_w, new_h))
    return scaled

def augment_image(image):
    # Apply random rotation
    angle = random.randint(-15, 15)
    rotated = rotate_image(image, angle)
    
    # Randomly flip the image
    if random.choice([True, False]):
        rotated = flip_image(rotated)
    
    # Randomly add noise
    if random.choice([True, False]):
        rotated = add_noise(rotated)
    
    return rotated
