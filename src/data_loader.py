"""
Data loading, preprocessing, and augmentation pipeline for PlantVillage dataset.
Uses OpenCV for image manipulation and Keras PyDataset for backend-agnostic data feeding.
"""

import os
import random
import cv2
import numpy as np
import keras

IMAGE_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 32


def preprocess_image_cv2(image_path_or_bytes, target_size=IMAGE_SIZE):
    """
    Load and preprocess a single image using OpenCV.
    - Reads in BGR format
    - Converts to RGB color space
    - Resizes to target dimension (e.g. 128x128)
    - Normalizes pixel values to [0.0, 1.0]
    """
    if isinstance(image_path_or_bytes, (str, bytes, bytearray)):
        if isinstance(image_path_or_bytes, str):
            if not os.path.exists(image_path_or_bytes):
                raise FileNotFoundError(f"Image not found at {image_path_or_bytes}")
            img_bgr = cv2.imread(image_path_or_bytes)
        else:
            nparr = np.frombuffer(image_path_or_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_path_or_bytes, np.ndarray):
        # Already an array; if 3 channels, check if RGB or BGR
        img_bgr = image_path_or_bytes
    else:
        raise ValueError("Unsupported image input type")

    if img_bgr is None:
        raise ValueError("Failed to decode image using OpenCV")

    # If RGB format passed in from PIL/Streamlit, keep RGB, else convert BGR->RGB
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr

    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized, img_resized


def get_opencv_visualizations(image_rgb):
    """
    Generate OpenCV visual representations of the leaf for analysis:
    - Grayscale
    - Canny Edge Detection (lesion & leaf boundary identification)
    - HSV Necrosis / Lesion Mask (highlighting brown/yellow diseased tissue)
    """
    # 1. Grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. Canny Edge Map
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 3. HSV Color Space Analysis for diseased spots (yellow/brown lesions)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # Brown / Yellow range in HSV
    lower_diseased = np.array([10, 40, 40])
    upper_diseased = np.array([35, 255, 255])
    lesion_mask = cv2.inRange(hsv, lower_diseased, upper_diseased)
    lesion_highlight = cv2.bitwise_and(image_rgb, image_rgb, mask=lesion_mask)
    
    return {
        "grayscale": gray,
        "edges": edges,
        "lesion_mask": lesion_mask,
        "lesion_highlight": lesion_highlight
    }


def augment_image_cv2(image):
    """
    Perform geometric and photometric data augmentation using OpenCV:
    - Random horizontal and vertical flips
    - Random rotation (-25 to +25 degrees)
    - Random brightness adjustment
    """
    augmented = image.copy()
    h, w = augmented.shape[:2]

    # Horizontal flip (50% probability)
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # Vertical flip (30% probability)
    if random.random() > 0.7:
        augmented = cv2.flip(augmented, 0)

    # Random rotation
    angle = random.uniform(-25.0, 25.0)
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented = cv2.warpAffine(augmented, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random brightness shift (photometric)
    brightness_factor = random.uniform(0.85, 1.15)
    augmented = np.clip(augmented * brightness_factor, 0.0, 1.0)

    return augmented


class PlantDataset(keras.utils.PyDataset):
    """
    PyDataset implementation for batch-wise loading, resizing, 
    normalizing, and augmenting plant images.
    """
    def __init__(self, file_paths, labels, batch_size=DEFAULT_BATCH_SIZE, 
                 img_size=IMAGE_SIZE, augment=False, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.file_paths = list(file_paths)
        self.labels = np.array(labels, dtype=np.int64)
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = self.labels[batch_idx]

        for i in batch_idx:
            path = self.file_paths[i]
            img_norm, _ = preprocess_image_cv2(path, target_size=self.img_size)
            if self.augment:
                img_norm = augment_image_cv2(img_norm)
            batch_images.append(img_norm)

        return np.array(batch_images, dtype=np.float32), batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_dataset_file_list(dataset_dir, max_samples_per_class=None, seed=42):
    """
    Scan dataset directory and return file paths, class indices, and class names.
    Optionally balance classes with max_samples_per_class.
    """
    random.seed(seed)
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_paths = []
    all_labels = []

    for name in class_names:
        class_dir = os.path.join(dataset_dir, name)
        valid_exts = ('.jpg', '.jpeg', '.png')
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(valid_exts)]
        random.shuffle(images)
        if max_samples_per_class is not None and len(images) > max_samples_per_class:
            images = images[:max_samples_per_class]

        for p in images:
            all_paths.append(p)
            all_labels.append(class_to_idx[name])

    return all_paths, all_labels, class_names, class_to_idx


def get_data_splits(dataset_dir, max_samples_per_class=500, train_ratio=0.8, batch_size=DEFAULT_BATCH_SIZE, seed=42):
    """
    Split the dataset into training and validation sets.
    Returns (train_dataset, val_dataset, class_names, class_to_idx).
    """
    all_paths, all_labels, class_names, class_to_idx = load_dataset_file_list(
        dataset_dir, max_samples_per_class=max_samples_per_class, seed=seed
    )

    combined = list(zip(all_paths, all_labels))
    random.seed(seed)
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)

    split_idx = int(len(all_paths) * train_ratio)
    train_paths, val_paths = all_paths[:split_idx], all_paths[split_idx:]
    train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]

    train_ds = PlantDataset(train_paths, train_labels, batch_size=batch_size, augment=True, shuffle=True)
    val_ds = PlantDataset(val_paths, val_labels, batch_size=batch_size, augment=False, shuffle=False)

    return train_ds, val_ds, class_names, class_to_idx
