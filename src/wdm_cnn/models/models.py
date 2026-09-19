import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import random

# --- Classification Models ---
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=5, in_channels=1):
        super().__init__()
        # Level 1
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        # Level 2
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        # Level 3
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        # Fully Connected
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64, 32)   
        self.fc2 = nn.Linear(32, n_classes)

    
    def forward(self, x):
        # Level 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        # Level 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        # Level 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        # Flatten and Fully Connected
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x) # (B, n_classes)
        
        return logits

class CNN5_CLS(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        # Block 1: 1 -> 32
        self.conv1a = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)   # 256 -> 128
        
        # Block 2: 32 -> 64
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)   # 128 -> 64
        
        # Block 3: 64 -> 128
        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)   # 64 -> 32
        
        # Block 4: 128 -> 256
        self.conv4a = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4b = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)   # 32 -> 16
        
        # Block 5: 256 -> 256
        self.conv5a = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5a = nn.BatchNorm2d(256)
        self.conv5b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5b = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2)   # 16 -> 8
        
        # Head
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(64, n_classes)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        # Block 4
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool4(x)
        # Block 5
        x = F.relu(self.bn5a(self.conv5a(x)))
        x = F.relu(self.bn5b(self.conv5b(x)))
        x = self.pool5(x)
        # Head
        x = F.adaptive_avg_pool2d(x, (1, 1))        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ConfigCNN(nn.Module):
    """
    Flexible CNN with configurable architecture.
    
    Parameters
    ----------
    n_classes : int (3 or 5)
    in_channels : int (default 1)
    n_blocks : int — number of conv blocks (2-5 typical)
    base_channels : int — channels in first block; doubles each block
    kernel_size : int (3 or 5)
    dropout : float
    activation : str — 'relu', 'leakyrelu', 'gelu', 'silu', 'elu'
    pool_type : str — 'max' or 'avg'
    use_bn : bool — BatchNorm after conv
    fc_dims : list — hidden FC layer dimensions (e.g., [32], [64, 32])
    """
    def __init__(self, n_classes=5, in_channels=1,
                 n_blocks=3, base_channels=16, kernel_size=3,
                 dropout=0.5, activation='relu', pool_type='max', 
                 pool_size=2, use_bn=True, fc_dims=[32]):
        super().__init__()

        act_map = {
            'relu': nn.ReLU, 'leakyrelu': nn.LeakyReLU,
            'gelu': nn.GELU, 'silu': nn.SiLU, 'elu': nn.ELU,
        }
        pool_map = {'max': nn.MaxPool2d, 'avg': nn.AvgPool2d}
        
        act = act_map[activation]
        pool = pool_map[pool_type]

        # Conv feature extractor
        layers = []
        c_in = in_channels
        for i in range(n_blocks):
            c_out = base_channels * (2**i)
            layers.append(nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size//2))
            if use_bn:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(act())
            layers.append(nn.Conv2d(c_out, c_out, kernel_size, padding=kernel_size//2))
            if use_bn:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(act())
            layers.append(pool(pool_size))
            c_in = c_out
        
        # Verify final spatial size is reasonable
        final_spatial = 256
        for _ in range(n_blocks):
            final_spatial //= pool_size
        if final_spatial < 1:
            raise ValueError(f'n_blocks={n_blocks} with pool_size={pool_size} reduces input to <1×1')
            
        self.features = nn.Sequential(*layers)
        self.final_channels = c_out

        # FC head
        fc_layers = []
        c_fc = c_out
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(c_fc, fc_dim))
            fc_layers.append(act())
            fc_layers.append(nn.Dropout(dropout))
            c_fc = fc_dim
        fc_layers.append(nn.Linear(c_fc, n_classes))
        self.classifier = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

# --- Dataset Classes ---
class RandomRot90:
    """Randomly rotate by 0, 90, 180, or 270 degrees."""
    def __call__(self, x):
        k = random.randint(0, 3)
        return torch.rot90(x, k, dims=(-2, -1))

class GalaxyDatasetOnlineNoise(Dataset):
    def __init__(self, noiseless_sb_data, labels, omega, sigma_sky, scaler, post_mean, post_std, augmentation=False, zp=0, rotation='random180'):
        """
        Parameters
        ----------
        noiseless_sb_data : np.ndarray of shape (N, H, W)
            Noiseless mock images in SB units (mag/arcsec²)
        labels : array-like of shape (N,)
            Class labels (integer)
        omega : float
            Pixel area from FITS header (PIXAREA) — same for all images
        sigma_sky : float
            Standard deviation of sky noise in flux units
        scaler : object with .transform(np.ndarray) method
            Fitted scaler (ShiftedLogScaler) applied per image
        augmentation : bool
            Whether to apply random rotation/flip transforms
        zp : float
            Photometric zero point for SB → flux conversion (default 0)
        """

        # Store noiseless SB data as float32 tensor with channel dim
        self.sb_data = torch.tensor(noiseless_sb_data, dtype=torch.float32)
        if self.sb_data.ndim == 3:
            self.sb_data = self.sb_data.unsqueeze(1)   # (N, H, W) → (N, 1, H, W)
        self.labels = torch.tensor(labels)
        self.omega = float(omega)
        self.sigma_sky = sigma_sky
        self.scaler = scaler
        self.zp = float(zp)
        self.post_mean=post_mean
        self.post_std = post_std

        if augmentation:
            transforms_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
            if rotation == 'random180':
                transforms_list.append(transforms.RandomRotation(180))
            elif rotation == 'rot90':
                transforms_list.append(RandomRot90())
            # 'none' → no rotation added
            
            self.transform = transforms.Compose(transforms_list)
        else:
            self.transform = None

    def __len__(self):
        return len(self.sb_data)

    def __getitem__(self, idx):
        sb_image = self.sb_data[idx]    # (1, H, W) in SB units
        label = self.labels[idx]
        # 1. Convert to flux units
        flux = self.omega * 10**(-(sb_image-self.zp)/2.5)

        # 2. Add Gaussian noise
        noise = torch.randn_like(flux) * self.sigma_sky
        flux_noisy = flux + noise

        # 3. Apply scaler (convert to numpy and back)
        flux_np = flux_noisy.numpy().squeeze(0) # (H,W)
        if isinstance(self.scaler, StandardScaler):
            # StandardScaler needs (n_samples, n_features) shape
            flux_scaled = self.scaler.transform(flux_np.reshape(-1, 1)).reshape(flux_np.shape)
        else:
            # Custom scalers (ShiftedLog, Arcsinh) accept (H, W) directly
            flux_scaled = self.scaler.transform(flux_np)
        image = torch.tensor(flux_scaled, dtype=torch.float32).unsqueeze(0)   # (1, H, W)
        # Standardize with pre-computed stats
        image = (image - self.post_mean) / self.post_std
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

class GalaxyDataset(Dataset):
    def __init__(self, data, labels, augmentation=False, rotation='random180'):
        data_tensor = torch.tensor(data, dtype=torch.float32)
    
        # Add channel dim only if data is 3D (single-channel images)
        if data_tensor.ndim == 3:
            data_tensor = data_tensor.unsqueeze(1)   # (N, H, W) -> (N, 1, H, W)
        # If 4D already (N, C, H, W), leave as is
        
        self.data = data_tensor
        self.labels = torch.tensor(labels)

        if augmentation:
            transforms_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
            if rotation == 'random180':
                transforms_list.append(transforms.RandomRotation(180))
            elif rotation == 'rot90':
                transforms_list.append(RandomRot90())
            # 'none' → no rotation added
            
            self.transform = transforms.Compose(transforms_list)
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]   # shape (1, H, W) or (3, H, W)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Regression Models ---
class RegrCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        # Level 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        # Level 2
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        # Level 3
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        # Fully Connected
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64, 32)   # after global avg pool
        self.fc2 = nn.Linear(32, 2)     # Maps the embedding to the mean value (WDM mass) and std

    
    def forward(self, x):
        # Level 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        # Level 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        # Level 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        # Flatten and Fully Connected
        x = F.adaptive_avg_pool2d(x, (1, 1))        # Unsure about Global Average Pooling, but works for any input size --> change (1,1) to (h,h) for hyperparam h 
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        out = self.fc2(x) # (B, 2)

        mean = out[:,0]
        log_std = out[:,1]        
        return mean, log_std   

