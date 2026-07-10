# GTA5→Cityscapes Data Loading Research

## 1. GTA5 Dataset Structure

### 1.1 Directory Layout

```
GTA5/
├── images/
│   ├── 00001.png    # 1914×1052 RGB images
│   ├── 00002.png
│   └── ...         # 24,966 total images
├── labels/
│   ├── 00001.png    # 1914×1052 RGB label maps (palette-encoded)
│   ├── 00002.png
│   └── ...         # 24,966 total labels
└── gta5_trainplits.mat  # .mat file defining train split
```

### 1.2 Reading the .mat Split File

The `.mat` file contains a MATLAB struct with a `names` field listing training image filenames.

```python
import scipy.io as sio
import numpy as np

# Load the .mat file
mat_data = sio.loadmat('gta5_trainplits.mat')

# The .mat file structure varies by version:
# Version 1: mat_data['name'] contains list of filenames
# Version 2: mat_data['train'] or mat_data['names'] contains indices

# Most common format (AdaptSegNet/DAFormer convention):
# File contains 'name' field as a numpy object array of strings
names = mat_data['name']
train_list = [str(names[i][0]).strip() for i in range(len(names))]

# Alternative format (some versions use indices):
# train_indices = mat_data['train'].flatten()  # 0-based indices
# train_list = [f'{int(idx):05d}.png' for idx in train_indices]
```

**Key point**: All 24,966 GTA5 images are used for training. The `.mat` file simply lists the filenames (sometimes with full paths). The standard UDA convention treats ALL GTA5 images as the training set.

### 1.3 Label Format: RGB Palette-Encoded Labels

GTA5 labels are **RGB PNG images** where each pixel's RGB color encodes the class ID. The raw GTA5 format uses 33 classes, but UDA research remaps to 19 Cityscapes-compatible classes.

#### GTA5 Raw RGB-to-ClassID Mapping (33 classes)

```python
# Raw GTA5 palette (RGB → class ID)
GTA5_PALETTE = {
    (0, 0, 0): 0,        # Road
    (0, 0, 0): 0,        # (duplicated in some versions)
    (0, 0, 111): 1,      # Sidewalk
    (0, 0, 142): 2,      # Building
    (0, 0, 230): 3,      # Wall
    (0, 119, 0): 4,      # Fence
    (0, 182, 0): 5,      # Pole
    (119, 11, 0): 6,     # Traffic Light
    (127, 63, 0): 7,     # Traffic Sign
    (0, 142, 0): 8,      # Vegetation
    (152, 251, 152): 9,  # Terrain
    (107, 142, 35): 10,  # Sky (note: swapped in some versions)
    (220, 20, 60): 11,   # Person
    (255, 0, 0): 12,     # Rider
    (0, 0, 70): 13,      # Car
    (0, 0, 142): 14,     # Truck (same as building in some!)
    (0, 60, 100): 15,    # Bus
    (0, 80, 100): 16,    # Train
    (0, 0, 230): 17,     # Motorcycle (same as wall!)
    (119, 11, 32): 18,   # Bicycle
    # ... additional classes
}
```

**Critical Issue**: The raw GTA5 palette has RGB collisions (e.g., Building and Car both map to (0, 0, 142)). The standard approach is to use a lookup table that processes each pixel's RGB value:

```python
def load_gta5_label(label_path, class_mapping):
    """Load GTA5 RGB label and convert to 19-class index map.
    
    Args:
        label_path: Path to RGB label PNG
        class_mapping: Dict mapping GTA5 class IDs to Cityscapes train IDs
    
    Returns:
        numpy array of shape (H, W) with values 0-18 (or 255 for void)
    """
    import cv2
    import numpy as np
    
    # Read as RGB (cv2 reads BGR, so convert)
    label_rgb = cv2.imread(label_path)[:, :, ::-1]  # BGR → RGB
    
    # Initialize with 255 (void/unlabeled)
    label_index = np.full(label_rgb.shape[:2], 255, dtype=np.uint8)
    
    # Map each unique RGB color to class ID
    unique_colors = np.unique(label_rgb.reshape(-1, 3), axis=0)
    
    for color in unique_colors:
        # Find corresponding GTA5 class ID
        r, g, b = color
        gta5_class = find_gta5_class(r, g, b)  # lookup function
        
        # Remap to Cityscapes 19-class format
        if gta5_class in class_mapping:
            cityscapes_id = class_mapping[gta5_class]
            mask = np.all(label_rgb == color, axis=-1)
            label_index[mask] = cityscapes_id
    
    return label_index
```

#### Standard GTA5 → Cityscapes 19-Class Mapping

```python
# GTA5 class ID → Cityscapes train_id
GTA5_TO_CITYSCAPES = {
    0: 0,    # Road
    1: 1,    # Sidewalk
    2: 2,    # Building
    3: 3,    # Wall
    4: 4,    # Fence
    5: 5,    # Pole
    6: 6,    # Traffic Light
    7: 7,    # Traffic Sign
    8: 8,    # Vegetation
    9: 9,    # Terrain
    10: 10,  # Sky
    11: 11,  # Person
    12: 12,  # Rider
    13: 13,  # Car
    14: 14,  # Truck
    15: 15,  # Bus
    16: 16,  # Train
    17: 17,  # Motorcycle
    18: 18,  # Bicycle
    # GTA5 classes 19-32 → 255 (void/unmapped)
}
```

#### Recommended Preprocessing Approach

The most robust approach is to **precompute index labels** before training:

```python
import os
import cv2
import numpy as np
from pathlib import Path
from scipy.io import loadmat

def preprocess_gta5_labels(gta5_root, output_dir):
    """Convert all GTA5 RGB labels to 19-class index maps."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load split file
    mat_data = loadmat(os.path.join(gta5_root, 'gta5_trainplits.mat'))
    names = [str(mat_data['name'][i][0]).strip() for i in range(len(mat_data['name']))]
    
    for name in names:
        label_path = os.path.join(gta5_root, 'labels', name)
        output_path = os.path.join(output_dir, name.replace('.png', '_label.png'))
        
        if os.path.exists(output_path):
            continue
        
        # Read RGB label
        label_rgb = cv2.imread(label_path)[:, :, ::-1]  # BGR → RGB
        
        # Convert to index map
        label_index = rgb_to_cityscapes_index(label_rgb)
        
        # Save as single-channel PNG (faster loading during training)
        cv2.imwrite(output_path, label_index)
    
    print(f"Preprocessed {len(names)} labels to {output_dir}")

def rgb_to_cityscapes_index(label_rgb):
    """Convert RGB label to 19-class index map."""
    h, w = label_rgb.shape[:2]
    label_index = np.full((h, w), 255, dtype=np.uint8)
    
    # Define complete mapping: (R, G, B) → train_id
    rgb_to_id = {
        (128, 64, 128): 0,    # road
        (244, 35, 232): 1,    # sidewalk
        (70, 70, 70): 2,      # building
        (102, 102, 156): 3,   # wall
        (190, 153, 153): 4,   # fence
        (153, 153, 153): 5,   # pole
        (250, 170, 30): 6,    # traffic light
        (220, 220, 0): 7,     # traffic sign
        (107, 142, 35): 8,    # vegetation
        (152, 251, 152): 9,   # terrain
        (70, 130, 180): 10,   # sky
        (220, 20, 60): 11,    # person
        (255, 0, 0): 12,      # rider
        (0, 0, 142): 13,      # car
        (0, 0, 70): 14,       # truck
        (0, 60, 100): 15,     # bus
        (0, 80, 100): 16,     # train
        (0, 0, 230): 17,      # motorcycle
        (119, 11, 32): 18,    # bicycle
    }
    
    for rgb, train_id in rgb_to_id.items():
        mask = np.all(label_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        label_index[mask] = train_id
    
    return label_index
```

---

## 2. Cityscapes Dataset Structure

### 2.1 Directory Layout

```
Cityscapes/
├── leftImg8bit/
│   ├── train/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_000019_leftImg8bit.png
│   │   │   ├── aachen_000001_000019_leftImg8bit.png
│   │   │   └── ...
│   │   ├── bochum/
│   │   └── ... (18 cities total in train)
│   └── val/
│       ├── frankfurt/
│       │   ├── frankfurt_000000_000216_leftImg8bit.png
│       │   └── ...
│       └── ... (5 cities total in val)
├── gtFine/
│   ├── train/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_000019_gtFine_color.png          # RGB visualization
│   │   │   ├── aachen_000000_000019_gtFine_instanceIds.png     # Instance segmentation
│   │   │   ├── aachen_000000_000019_gtFine_labelIds.png        # 34 class IDs (raw)
│   │   │   ├── aachen_000000_000019_gtFine_labelTrainIds.png   # 19 class IDs (train)
│   │   │   └── aachen_000000_000019_gtFine_polygons.json       # Polygon annotations
│   │   └── ...
│   └── val/
│       ├── frankfurt/
│       │   └── ...
│       └── ...
└── cityscapes_leftImg8bit_trainextra/  # Optional: extra training data (no labels)
```

### 2.2 Image Naming Convention

```
{city}_{seq_id:06d}_{frame_id:06d}_leftImg8bit.png
{city}_{seq_id:06d}_{frame_id:06d}_gtFine_{annotation_type}.png
```

Example: `aachen_000000_000019_leftImg8bit.png` and `aachen_000000_000019_gtFine_labelTrainIds.png`

### 2.3 Key Annotation Files

| File | Description | Values |
|------|-------------|--------|
| `*_gtFine_labelTrainIds.png` | **19-class training format** | 0-18 for valid classes, 255 for void |
| `*_gtFine_labelIds.png` | Raw 34 class IDs | 0-33 (includes invalid/ignored classes) |
| `*_gtFine_color.png` | RGB visualization | Palette-encoded for visualization only |
| `*_gtFine_instanceIds.png` | Instance IDs | Unique ID per object instance |
| `*_gtFine_polygons.json` | Vector annotations | Polygon coordinates |

### 2.4 Reading gtFine_labelTrainIds.png

This is the **standard format for training**. The PNG file is a single-channel image where each pixel value is the 19-class train ID.

```python
import cv2
import numpy as np

def load_cityscapes_label(label_path):
    """Load Cityscapes labelTrainIds.png.
    
    Returns:
        numpy array of shape (H, W) with values 0-18 or 255
    """
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    return label  # Already in train_id format
```

**Important**: The `labelTrainIds.png` files are NOT included in the default Cityscapes download. You must run the Cityscapes preparation script (`cityscapesscripts/preparation/_createTrainIdLabelImgs.py`) to generate them.

```bash
# Generate labelTrainIds from polygons
python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

### 2.5 Cityscapes 19-Class Train IDs

```python
CITYSCAPES_CLASSES = {
    0: 'road',
    1: 'sidewalk', 
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
    255: 'void'  # Ignore index for evaluation
}

# RGB palette for visualization
CITYSCAPES_PALETTE = {
    0: (128, 64, 128),    # road
    1: (244, 35, 232),    # sidewalk
    2: (70, 70, 70),      # building
    3: (102, 102, 156),   # wall
    4: (190, 153, 153),   # fence
    5: (153, 153, 153),   # pole
    6: (250, 170, 30),    # traffic light
    7: (220, 220, 0),     # traffic sign
    8: (107, 142, 35),    # vegetation
    9: (152, 251, 152),   # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
}
```

---

## 3. GTA5 Source Augmentation (Training Images)

### 3.1 Standard Augmentation Parameters from UDA Papers

Based on DAFormer, MIC, HRDA, and AdaptSegNet:

| Augmentation | Parameter | Value | Notes |
|--------------|-----------|-------|-------|
| Random Resize | scale range | **0.5–2.0×** | Rescale image randomly |
| Random Crop | crop size | **512×512** | Center crop if image smaller |
| Random Horizontal Flip | probability | **0.5** | Standard flip probability |
| Color Jitter | brightness | **0.4** | Max brightness change |
| Color Jitter | contrast | **0.4** | Max contrast change |
| Color Jitter | saturation | **0.4** | Max saturation change |
| Color Jitter | hue | **0.1** | Max hue change (small) |
| Normalize | mean | [0.485, 0.456, 0.406] | ImageNet RGB mean |
| Normalize | std | [0.229, 0.224, 0.225] | ImageNet RGB std |

### 3.2 Complete GTA5 Augmentation Pipeline

```python
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

class GTA5Augmentation:
    """Standard augmentation for GTA5 source images."""
    
    def __init__(self, crop_size=512, resize_range=(0.5, 2.0)):
        self.crop_size = crop_size
        self.resize_range = resize_range
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, image, label):
        """
        Args:
            image: PIL Image or tensor (C, H, W)
            label: PIL Image or tensor (H, W) with class indices
        
        Returns:
            Augmented image and label
        """
        # Random resize
        scale = random.uniform(*self.resize_range)
        new_h, new_w = int(image.height * scale), int(image.width * scale)
        image = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)
        
        # Random crop
        i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)
        
        # Random horizontal flip
        if random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)
        
        # Color jitter (applied to image only)
        color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        image = color_jitter(image)
        
        # To tensor and normalize
        image = TF.to_tensor(image)
        image = self.normalize(image)
        
        # Label to tensor (long)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
```

### 3.3 ImageNet Normalization Values

```python
# Standard ImageNet normalization (used in most UDA papers)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB

# Some papers use pixel-scale normalization (0-255 range):
# mean = [123.675, 116.28, 103.53]
# std = [58.395, 57.12, 57.375]
```

---

## 4. Cityscapes Target Augmentation (Teacher-Student UDA)

### 4.1 Key Difference from Source

In teacher-student UDA, **target images receive spatial-only augmentation** (no color jitter) to ensure consistency between teacher and student predictions. The teacher sees the target image with spatial transforms only.

### 4.2 Standard Target Augmentation Parameters

| Augmentation | Parameter | Value | Notes |
|--------------|-----------|-------|-------|
| Random Resize | scale range | **0.5–2.0×** | Same as source |
| Random Crop | crop size | **512×512** | Same as source |
| Random Horizontal Flip | probability | **0.5** | Same as source |
| Color Jitter | **None** | **None** | **Not applied to target** |
| Normalize | mean | [0.485, 0.456, 0.406] | Same ImageNet normalization |
| Normalize | std | [0.229, 0.224, 0.225] | Same ImageNet normalization |

### 4.3 Complete Cityscapes Target Augmentation Pipeline

```python
class CityscapesTargetAugmentation:
    """Spatial-only augmentation for Cityscapes target images.
    
    No color jitter is applied to ensure consistency between
    teacher and student predictions in the UDA framework.
    """
    
    def __init__(self, crop_size=512, resize_range=(0.5, 2.0)):
        self.crop_size = crop_size
        self.resize_range = resize_range
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, image, heatmap=None):
        """
        Args:
            image: PIL Image (target domain image)
            heatmap: Optional precomputed heatmap (H, W) for SATG
        
        Returns:
            Augmented image and heatmap (if provided)
        """
        # Random resize
        scale = random.uniform(*self.resize_range)
        new_h, new_w = int(image.height * scale), int(image.width * scale)
        image = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        
        if heatmap is not None:
            heatmap = TF.resize(heatmap, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        
        # Random crop
        i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        
        if heatmap is not None:
            heatmap = heatmap[i:i+h, j:j+w]
        
        # Random horizontal flip
        if random.random() < 0.5:
            image = TF.hflip(image)
            if heatmap is not None:
                heatmap = TF.hflip(heatmap)
        
        # To tensor and normalize
        image = TF.to_tensor(image)
        image = self.normalize(image)
        
        if heatmap is not None:
            heatmap = torch.tensor(heatmap, dtype=torch.float32)
        
        return image, heatmap
```

---

## 5. PyTorch DataLoader Pattern for Dual Datasets

### 5.1 The Problem: Different Dataset Sizes

- GTA5 training: **24,966 images**
- Cityscapes training: **2,975 images**

The Cityscapes dataset is ~8.4× smaller. We need a strategy to handle this mismatch.

### 5.2 Recommended Pattern: Cycling the Shorter Dataset

The standard approach in UDA research is to **cycle the shorter dataset** (Cityscapes) so it repeats until the longer dataset (GTA5) is exhausted.

```python
import itertools
from torch.utils.data import DataLoader, Dataset

class DualDomainDataLoader:
    """Manages source and target DataLoaders for UDA training.
    
    Standard pattern: cycle the shorter dataset (target/Cityscapes)
    to match the longer dataset (source/GTA5).
    """
    
    def __init__(self, source_dataset, target_dataset, 
                 source_batch_size=1, target_batch_size=1,
                 num_workers=4):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        
        # Source loader (longer dataset - GTA5)
        self.source_loader = DataLoader(
            source_dataset,
            batch_size=source_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Target loader (shorter dataset - Cityscapes, with cycling)
        self.target_loader = DataLoader(
            target_dataset,
            batch_size=target_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Create infinite iterator by cycling
        self.target_iter = itertools.cycle(self.target_loader)
    
    def __iter__(self):
        """Yield (source_batch, target_batch) pairs."""
        for source_batch in self.source_loader:
            target_batch = next(self.target_iter)
            yield source_batch, target_batch
    
    def __len__(self):
        """Length equals the source loader (longer dataset)."""
        return len(self.source_loader)
```

### 5.3 Alternative Pattern: Zip with Repeat

```python
def create_infinite_loader(dataset, batch_size, num_workers=4):
    """Create an infinite data loader by repeating the dataset."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return itertools.cycle(loader)

# In training loop
source_loader = DataLoader(source_dataset, batch_size=1, shuffle=True, num_workers=4)
target_iter = create_infinite_loader(target_dataset, batch_size=1)

for source_batch in source_loader:
    target_batch = next(target_iter)
    # ... training step
```

### 5.4 PyTorch IterableDataset Pattern (Advanced)

For more sophisticated handling, use `IterableDataset`:

```python
from torch.utils.data import IterableDataset, get_worker_info

class InfiniteSampler(IterableDataset):
    """Infinite sampler that cycles through a dataset."""
    
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
    
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Single worker
            indices = list(range(len(self.dataset)))
        else:
            # Multiple workers: split indices
            per_worker = len(self.dataset) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker
            indices = list(range(start, end))
        
        while True:
            if self.shuffle:
                random.shuffle(indices)
            for idx in indices:
                yield self.dataset[idx]
```

### 5.5 Complete Training Loop Pattern

```python
def train_uda(source_dataset, target_dataset, teacher_model, student_model, 
              optimizer, device, num_iterations=40000):
    """Complete UDA training loop with dual DataLoaders."""
    
    # Create dual dataloader
    dual_loader = DualDomainDataLoader(
        source_dataset, 
        target_dataset,
        source_batch_size=1,
        target_batch_size=1,
        num_workers=4
    )
    
    # EMA teacher
    ema_momentum = 0.999
    
    for iteration, (source_batch, target_batch) in enumerate(dual_loader):
        if iteration >= num_iterations:
            break
        
        # Unpack batches
        source_images, source_labels = source_batch
        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        
        target_images, target_heatmaps = target_batch
        target_images = target_images.to(device)
        target_heatmaps = target_heatmaps.to(device)
        
        # Forward pass
        student_model.train()
        source_logits = student_model(source_images)
        target_logits = student_model(target_images)
        
        # Source loss (cross-entropy with labels)
        source_loss = F.cross_entropy(source_logits, source_labels, ignore_index=255)
        
        # Target loss (pseudo-labels with trust gating)
        with torch.no_grad():
            teacher_logits = teacher_model(target_images)
            pseudo_labels = teacher_logits.argmax(dim=1)
            confidence = teacher_logits.softmax(dim=1).max(dim=1)[0]
            
            # SATG trust gate
            trust_mask = (confidence > 0.9) & (target_heatmaps < 0.6)
        
        # Target loss with trust mask
        target_loss = F.cross_entropy(target_logits, pseudo_labels, reduction='none')
        target_loss = (target_loss * trust_mask.float()).mean()
        
        # Total loss
        total_loss = source_loss + target_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # EMA update
        with torch.no_grad():
            for teacher_param, student_param in zip(teacher_model.parameters(), 
                                                    student_model.parameters()):
                teacher_param.data = ema_momentum * teacher_param.data + \
                                    (1 - ema_momentum) * student_param.data
        
        # Logging
        if iteration % 100 == 0:
            trust_ratio = trust_mask.float().mean().item()
            print(f"Iter {iteration}: loss={total_loss.item():.4f}, "
                  f"src={source_loss.item():.4f}, tgt={target_loss.item():.4f}, "
                  f"trust={trust_ratio:.4f}")
    
    return teacher_model
```

### 5.6 Alternative: Two Separate Iterators

```python
def train_with_separate_iterators(source_dataset, target_dataset, num_workers=4):
    """Alternative pattern using two separate iterators."""
    
    source_loader = DataLoader(
        source_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Create infinite target iterator
    target_iter = iter(target_loader)
    
    for source_batch in source_loader:
        # Reset target iterator if exhausted
        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)
        
        yield source_batch, target_batch
```

---

## 6. Recommended Implementation Pattern

### 6.1 Dataset Class Structure

```python
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np

class GTA5Dataset(Dataset):
    """GTA5 source dataset with precomputed index labels."""
    
    def __init__(self, root, split='train', transform=None):
        self.root = Path(root)
        self.transform = transform
        
        # Load split
        import scipy.io as sio
        mat_data = sio.loadmat(self.root / 'gta5_trainplits.mat')
        self.image_names = [str(mat_data['name'][i][0]).strip() 
                           for i in range(len(mat_data['name']))]
        
        # Use precomputed index labels
        self.label_dir = self.root / 'labels_index'
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        name = self.image_names[idx]
        
        # Load image
        image_path = self.root / 'images' / name
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Load precomputed index label
        label_path = self.label_dir / name
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        label = Image.fromarray(label)
        
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label


class CityscapesDataset(Dataset):
    """Cityscapes target dataset with precomputed heatmaps."""
    
    def __init__(self, root, split='train', transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Collect images
        self.images = []
        img_dir = self.root / 'leftImg8bit' / split
        
        for city_dir in sorted(img_dir.iterdir()):
            if city_dir.is_dir():
                for img_path in sorted(city_dir.glob('*_leftImg8bit.png')):
                    self.images.append(img_path)
        
        # Heatmap directory
        self.heatmap_dir = self.root / 'heatmaps' / split
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Load precomputed heatmap (4 components combined at load time)
        stem = img_path.stem.replace('_leftImg8bit', '')
        comps = {}
        for key in ('edge', 'var', 'ent', 'corn'):
            p = self.heatmap_dir / f"{stem}_satg_{key}.npy"
            comps[key] = np.load(str(p))
        heatmap = 0.25 * sum(comps.values())  # weighted combination from config
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = torch.from_numpy(heatmap).float()
        
        if self.transform:
            image, heatmap = self.transform(image, heatmap)
        
        return image, heatmap
```

### 6.2 DataLoader Configuration

```python
# Recommended DataLoader settings
source_loader = DataLoader(
    GTA5Dataset(root='/Users/aadisaraf/Documents/research/SATG/data/GTA5', transform=gta5_transform),
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True  # Keeps workers alive between epochs
)

target_loader = DataLoader(
    CityscapesDataset(root='data/Cityscapes', transform=target_transform),
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True
)

# Cycle target loader to match source length
target_iter = itertools.cycle(target_loader)

for source_batch in source_loader:
    target_batch = next(target_iter)
    # ... training step
```

---

## 7. Key Implementation Notes

### 7.1 Label Encoding

| Dataset | Label Format | Values | File Type |
|---------|-------------|--------|-----------|
| GTA5 (raw) | RGB palette | (R,G,B) colors | 3-channel PNG |
| GTA5 (preprocessed) | Index map | 0-18, 255 | Single-channel PNG |
| Cityscapes (labelTrainIds) | Index map | 0-18, 255 | Single-channel PNG |

### 7.2 Performance Optimization

1. **Precompute index labels** for GTA5 (convert RGB → index once)
2. **Precompute heatmaps** for Cityscapes (avoid recomputation during training)
3. **Use `persistent_workers=True`** to avoid worker respawn overhead
4. **Use `pin_memory=True`** for faster CPU→GPU transfer
5. **Use `drop_last=True`** to avoid incomplete batches

### 7.3 Common Pitfalls

1. **GTA5 label RGB collisions**: Use preprocessed index labels, not raw RGB
2. **Cityscapes labelTrainIds missing**: Run preparation script first
3. **Augmentation mismatch**: Target images must NOT receive color jitter
4. **Heatmap augmentation**: Apply identical spatial transforms to image and heatmap
5. **Label ignore index**: Use `ignore_index=255` in cross-entropy loss

### 7.4 Standard Hyperparameters Summary

```python
# GTA5 source augmentation
SOURCE_AUGMENTATION = {
    'resize_range': (0.5, 2.0),
    'crop_size': 512,
    'flip_prob': 0.5,
    'color_jitter': {
        'brightness': 0.4,
        'contrast': 0.4,
        'saturation': 0.4,
        'hue': 0.1
    },
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225]
}

# Cityscapes target augmentation (spatial only)
TARGET_AUGMENTATION = {
    'resize_range': (0.5, 2.0),
    'crop_size': 512,
    'flip_prob': 0.5,
    'color_jitter': None,  # No color augmentation
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225]
}

# DataLoader settings
DATALOADER_CONFIG = {
    'batch_size': 1,  # One source + one target per GPU
    'num_workers': 4,
    'pin_memory': True,
    'drop_last': True,
    'persistent_workers': True
}
```
