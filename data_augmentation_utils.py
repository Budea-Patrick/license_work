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
    return opencv.flip(image, 1) 

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
    
    # Randomly add noise
    if random.choice([True, False]):
        rotated = add_noise(rotated)

    # Apply random scaling
    scale_factor = random.uniform(0.8, 1.2)
    scaled = scale_image(rotated, scale_factor)
    
    return scaled

def rotate_landmarks(landmarks, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    rotated_landmarks = []
    for i in range(0, len(landmarks), 3):
        x, y, z = landmarks[i], landmarks[i + 1], landmarks[i + 2]
        rotated_point = np.dot(rotation_matrix, np.array([x, y]))
        rotated_landmarks.extend([rotated_point[0], rotated_point[1], z])
    return np.array(rotated_landmarks)

def scale_landmarks(landmarks, scale_factor):
    return landmarks * scale_factor

def add_noise_to_landmarks(landmarks, noise_level=0.01):
    noise = np.random.normal(0, noise_level, landmarks.shape)
    return landmarks + noise

def mirror_landmarks(landmarks, image_width):
    mirrored_landmarks = []
    for i in range(0, len(landmarks), 3):
        x, y, z = landmarks[i], landmarks[i + 1], landmarks[i + 2]
        mirrored_landmarks.extend([image_width - x, y, z])
    return np.array(mirrored_landmarks)

def augment_data(data, image_width):
    augmented_data = []
    for landmarks, label in data:
        augmented_data.append((landmarks, label))
        
        # Apply rotation
        for angle in [-15, 0, 15]:
            rotated_landmarks = rotate_landmarks(landmarks, angle)
            augmented_data.append((rotated_landmarks, label))
        
        # Apply scaling
        for scale_factor in [0.9, 1.0, 1.1]:
            scaled_landmarks = scale_landmarks(landmarks, scale_factor)
            augmented_data.append((scaled_landmarks, label))
        
        # Apply noise
        noisy_landmarks = add_noise_to_landmarks(landmarks)
        augmented_data.append((noisy_landmarks, label))
        
        # Apply mirroring
        mirrored_landmarks = mirror_landmarks(landmarks, image_width)
        augmented_data.append((mirrored_landmarks, label))
        
    return augmented_data
