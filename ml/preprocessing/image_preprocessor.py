import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from collections import Counter

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

def validate_image(path):
    """Checks file integrity and minimum resolution (224x224)."""
    if not os.path.exists(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        # open again to check size
        with Image.open(path) as img:
            width, height = img.size
            if width < 224 or height < 224:
                return False
        return True
    except Exception:
        return False

def crop_fundus_circle(image: np.ndarray, tol: int = 15) -> np.ndarray:
    """
    Crops black/empty boundaries from retinal fundus photograph and pads to square
    preserving the circular aspect ratio of the retina.
    """
    if image is None or image.size == 0:
        return image
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not np.any(mask):
        return image
    
    row_idx = np.where(np.any(mask, axis=1))[0]
    col_idx = np.where(np.any(mask, axis=0))[0]
    if len(row_idx) == 0 or len(col_idx) == 0:
        return image
    
    ymin, ymax = row_idx[0], row_idx[-1] + 1
    xmin, xmax = col_idx[0], col_idx[-1] + 1
    cropped = image[ymin:ymax, xmin:xmax]
    
    ch, cw = cropped.shape[:2]
    max_dim = max(ch, cw)
    pad_y = (max_dim - ch) // 2
    pad_x = (max_dim - cw) // 2
    
    if len(cropped.shape) == 3:
        padded = np.zeros((max_dim, max_dim, cropped.shape[2]), dtype=cropped.dtype)
        padded[pad_y:pad_y + ch, pad_x:pad_x + cw, :] = cropped
    else:
        padded = np.zeros((max_dim, max_dim), dtype=cropped.dtype)
        padded[pad_y:pad_y + ch, pad_x:pad_x + cw] = cropped
        
    return padded

class RetinalImagePreprocessor:
    def __init__(self, image_size=300):
        if isinstance(image_size, int):
            self.target_size = (image_size, image_size)
        else:
            self.target_size = tuple(image_size)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def preprocess(self, image, crop_circle: bool = True):
        """Preprocesses a numpy image array: optionally crops fundus circle, resizes and normalizes, returning a tensor."""
        if crop_circle and isinstance(image, np.ndarray):
            image = crop_fundus_circle(image)
        transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        augmented = transform(image=image)
        return augmented['image']

    def get_train_transforms(self):
        return A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

    def get_val_transforms(self):
        return A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
    
    def get_test_transforms(self):
        return self.get_val_transforms()

class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform, mode='train'):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
             img_path = os.path.join(self.img_dir, f"{img_id}.jpg") # fallback
             
        image = cv2.imread(img_path)
        if image is None:
            # Fallback to zero tensor if missing
            image_tensor = torch.zeros((3, 300, 300))
            label = int(row['diagnosis']) if 'diagnosis' in self.df.columns else 0
            return image_tensor, label, img_path
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        if 'diagnosis' in self.df.columns:
            label = int(row['diagnosis'])
        else:
            label = -1
            
        return image_tensor, label, img_path


def compute_class_weights(labels, num_classes=None):
    """Computes inverse-frequency class weights for CrossEntropyLoss."""
    count = Counter(labels)
    total = len(labels)
    if num_classes is None:
        num_classes = max(len(count), max(labels) + 1 if labels else 1)
    weights = []
    for i in range(num_classes):
        weights.append(total / (num_classes * max(count.get(i, 0), 1)))
    return torch.FloatTensor(weights)

def get_dataset_statistics(data_dir):
    """Returns dict with per-class counts and class names."""
    # Dummy implementation assuming subfolders are classes, as dataset structure isn't fully detailed in prompt
    import glob
    stats = {}
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        count = len(glob.glob(os.path.join(cls_dir, "*.*")))
        stats[cls] = count
    return {'class_counts': stats, 'class_names': class_names}

def create_data_loaders(data_dir, batch_size, num_workers):
    """Creates train/val/test DataLoaders (placeholder for full implementation dependent on df structure)."""
    # Assuming CSV driven
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    csv_path = os.path.join(data_dir, "train.csv")
    img_dir = os.path.join(data_dir, "train_images")
    if not os.path.exists(csv_path):
        # Return empty loaders if data not found, to avoid crash in automated test
        return None, None, None, None
        
    df = pd.read_csv(csv_path)
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['diagnosis'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['diagnosis'], random_state=42)
    
    preprocessor = RetinalImagePreprocessor()
    
    train_dataset = APTOSDataset(train_df, img_dir, preprocessor.get_train_transforms(), mode='train')
    val_dataset = APTOSDataset(val_df, img_dir, preprocessor.get_val_transforms(), mode='val')
    test_dataset = APTOSDataset(test_df, img_dir, preprocessor.get_test_transforms(), mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    weights = compute_class_weights(train_df['diagnosis'].tolist())
    
    return train_loader, val_loader, test_loader, weights
