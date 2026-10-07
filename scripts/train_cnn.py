# train.py
import sys
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from wdm_cnn.models.models import SimpleCNN, GalaxyDataset, GalaxyDatasetOnlineNoise, CNN5_CLS
from wdm_cnn.preprocessing.scalers import PowerLogScaler, SinhLogScaler, ArcsinhScaler, ShiftedLogScaler
from wdm_cnn.data.io import get_data 
from wdm_cnn.data.loaders import dm_models_for, _build_classification_suffix
from wdm_cnn.data.labels import label_str2int
from wdm_cnn.data.photometry import get_flux
from wdm_cnn.preprocessing.multiband import make_3channel_scaler_input
from config import UZH_THESIS, UZH_ROOT, ROT_TAG_MAP
import argparse
from astropy.io import fits
from pNbody.Mockimgs import noise
from pathlib import Path
import pickle

# --- Parser ---
parser = argparse.ArgumentParser(description='Train CNN with different scalers')
parser.add_argument('--scaler', type=str,
                    default = 'standard',
                    choices = ['standard', 'powerlog', 'sinhlog', 'arcsinh', 'shiftedlog', 'allscalers'],
                    help = 'Normalization scaler to use')

parser.add_argument('--augment', action='store_true', default=False,
                    help='Apply online data augmentation')

parser.add_argument('--exclude', action='store_true', default=False,
                    help='Exclude specific (halo, dm) pairs')

parser.add_argument('--SB_Limit', type=int,
                   default=32, help='Surface brightness limit of the SB maps to use')

parser.add_argument('--snapshots', type=str,
                    default='100', choices=['100', 'all'],
                    help='Use Redshift z=0 (snapshot=100) or all avalable redshifts.')

parser.add_argument('--model', default='SimpleCNN',
                    help='Choose NN model')

parser.add_argument('--n_classes', type=int,
                    default=5, choices=[3,5],
                    help='Number of Classes to use.')

parser.add_argument('--online_noise', action='store_true', default=False,
                    help='Use online noise addition instead of pre-noised images')

parser.add_argument('--split_version', type=str, default='v1',
                    help='Halo split version (v1, v2, ...)')

parser.add_argument('--rotation', type=str, default='random180',
                    choices=['random180', 'rot90', 'none'],
                    help='Rotation augmentation style')

parser.add_argument('--filter', type=str, default='VIS', 
                    choices=['VIS', 'H', 'J', 'Y', 'all'],
                    help='Filter band')

opt = parser.parse_args()

print(f'Using scaler: {opt.scaler}')


# --- Config ---
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

dm_models = dm_models_for(opt.n_classes)

if opt.snapshots=='100':
    snaps=[100]
elif opt.snapshots=='all':
    from config import DEFAULT_SNAPSHOTS
    snaps=DEFAULT_SNAPSHOTS
else:
    raise ValueError(f"Unknown snapshots value. Choose between '100' for z=0 or 'all' for all z " )


if opt.exclude:
    from config import EXCLUDED_PAIRS
    excluded = EXCLUDED_PAIRS
else: 
    excluded = []


# --- Load Data ---

if opt.split_version == 'v1':
    train_halos = np.load(f'{UZH_THESIS}/train_halos.npy')
    val_halos = np.load(f'{UZH_THESIS}/val_halos.npy')
else:
    train_halos = np.load(f'{UZH_THESIS}/train_halos_{opt.split_version}.npy')
    val_halos = np.load(f'{UZH_THESIS}/val_halos_{opt.split_version}.npy')

filter_app = f'_{opt.filter}' if opt.filter != 'VIS' else ''

if opt.online_noise:
    data_path = UZH_ROOT / f"input_images{filter_app}/"
    N_appendix = ""
else:
    data_path = UZH_ROOT / f"input_images{filter_app}_{opt.SB_Limit}_f/"
    N_appendix = f"_N{opt.SB_Limit}"


# Consider using the argument 'exclude=[(51, 'wdm03d5'), (51, 'wdm05')]' 
train_data, train_labels, train_files = get_data(data_path, train_halos, dm_models, snaps, False, exclude=excluded)   
val_data, val_labels, val_files = get_data(data_path, val_halos, dm_models, snaps, False, exclude=excluded)

train_labels_int = label_str2int(train_labels, dm_models)
val_labels_int = label_str2int(val_labels, dm_models)


# --- Scaler & Dataset Construction ---
if opt.online_noise:
    # ======= Online noise path =======
    # Read header info from one file
    example_file = f"{data_path}/FB15_h15_L11_r4_p1/100/SB_h15_cdm_100_LOS00{N_appendix}.fits.gz"
    with fits.open(example_file) as hdul:
        pixfovx = hdul[0].header['PIXFOVX']
        omega = hdul[0].header['PIXAREA']
    
    sigma_sky = noise.get_std_for_SB_limit(opt.SB_Limit, pixfovx, area=100, sn=3)
    print(f'Online noise: omega={omega}, pixfovx={pixfovx}, sigma_sky={sigma_sky:.4e}')


    # Create scaler
    if opt.scaler == 'shiftedlog':
        scaler = ShiftedLogScaler(shift=10*sigma_sky)
        scaler_str= 'shl'
    elif opt.scaler == 'arcsinh':
        scaler = ArcsinhScaler(soft=sigma_sky)
        scaler_str = 'as'
    else:
        raise ValueError(f"Online noise not supported for scaler: {opt.scaler}")
        
    stats_cache_path = f'{UZH_THESIS}/Losses/post_stretch_stats_{opt.model}_{scaler_str}_SBL{opt.SB_Limit}_z{opt.snapshots}_n{opt.n_classes}{"_aug" if opt.augment else ""}{"_excl" if opt.exclude else ""}{"_online" if opt.online_noise else ""}{"_"+opt.split_version if opt.split_version!="v1" else ""}.npy'
    if os.path.exists(stats_cache_path):
        print(f'Loading cached post-stretch stats from {stats_cache_path}')
        stats = np.load(stats_cache_path)
        post_mean = float(stats[0])
        post_std = float(stats[1])
    else:
        rng = np.random.default_rng(seed=42)
        n_samples = min(500, len(train_data))
        sample_idx = rng.choice(len(train_data), n_samples, replace=False)
        all_scaled = []
        for idx in sample_idx:
            sb = train_data[idx]
            flux = get_flux(sb, omega)
            n = rng.normal(0, sigma_sky, size=flux.shape)
            flux_noisy = flux + n
            if isinstance(scaler, StandardScaler):
                scaled = scaler.transform(flux_noisy.reshape(-1, 1)).reshape(flux_noisy.shape)
            else:
                scaled = scaler.transform(flux_noisy)
            all_scaled.append(scaled.flatten())
        
        all_scaled = np.concatenate(all_scaled)
        post_mean = float(all_scaled.mean())
        post_std = float(all_scaled.std())
        np.save(stats_cache_path, np.array([post_mean, post_std]))
    print(f'Post-stretch stats: mean={post_mean:.4f}, std={post_std:.4f}')
    
    
    # Build datasets
    train_set = GalaxyDatasetOnlineNoise(
        train_data, train_labels_int,
        omega=omega, sigma_sky=sigma_sky, scaler=scaler,
        post_mean=post_mean, post_std=post_std,
        augmentation=opt.augment, zp=0, rotation=opt.rotation
    )
    val_set = GalaxyDatasetOnlineNoise(
        val_data, val_labels_int,
        omega=omega, sigma_sky=sigma_sky, scaler=scaler,
        post_mean=post_mean, post_std=post_std,
        augmentation=False, zp=0, rotation=opt.rotation
    )
    
else:
    # === Offline Path ===
    # --- Preprocessing ---
    # Standardization

    filter_app = f'_{opt.filter}' if opt.filter != 'VIS' else ''
    img_path = UZH_ROOT / Path(f"input_images{filter_app}_{opt.SB_Limit}_f/FB15_h15_L11_r4_p1/100/SB_h15_cdm_100_LOS00_N{opt.SB_Limit}.fits.gz")
    sigma_sky = fits.open(img_path)[0].header['NOISESTD']
    
    print(f'\n=== Diagnostic info ===')
    print(f'Filter: {opt.filter}')
    print(f'img_path used for sigma_sky: {img_path}')
    print(f'sigma_sky: {sigma_sky:.4e}')
    print(f'5*sigma_sky (shift): {5*sigma_sky:.4e}')

    print(f'\ntrain_data:')
    print(f'  shape: {train_data.shape}')
    print(f'  range: [{train_data.min():.4e}, {train_data.max():.4e}]')
    print(f'  median: {np.median(train_data):.4e}')
    median_flux = np.median(train_data[train_data > 0])
    print(f'  median nonzero: {np.median(train_data[train_data > 0]):.4e}')
    print(f'  fraction nonzero: {(train_data != 0).mean():.4f}')

    # Apply different Normalizations (aka different scalers):
    if opt.scaler == 'standard':
        scaler = StandardScaler()
        scaler.fit(train_data.reshape(-1,1))
        train_data_norm = scaler.transform(train_data.reshape(-1,1)).reshape(train_data.shape)
        val_data_norm = scaler.transform(val_data.reshape(-1,1)).reshape(val_data.shape)
        scaler_str = 'std'
    elif opt.scaler == 'powerlog':
        scaler = PowerLogScaler(k=1.0, x0=median_flux, q=1.0)
        train_data_norm = scaler.transform(train_data)
        val_data_norm = scaler.transform(val_data)
        scaler_str = 'pl'
    elif opt.scaler == 'sinhlog':
        scaler = SinhLogScaler(alpha=1/sigma_sky, beta=3.0)
        train_data_norm = scaler.transform(train_data)
        val_data_norm = scaler.transform(val_data)
        scaler_str = 'sl'
    elif opt.scaler == 'arcsinh':
        scaler = ArcsinhScaler(soft=sigma_sky)
        train_data_norm = scaler.transform(train_data)
        val_data_norm   = scaler.transform(val_data)
        scaler_str = 'as'
    elif opt.scaler == 'shiftedlog':
        scaler = ShiftedLogScaler(shift=5*sigma_sky)
        train_data_norm = scaler.transform(train_data)
        val_data_norm   = scaler.transform(val_data)
        # test_data_norm  = scaler.transform(test_data)
        scaler_str = 'shl'
    elif opt.scaler == 'allscalers':
        scaler_shl = ShiftedLogScaler(shift=5*sigma_sky)
        scaler_as = ArcsinhScaler(soft=sigma_sky)
        scaler_std = StandardScaler()
        scaler_std.fit(train_data.reshape(-1,1))
        train_data_norm = make_3channel_scaler_input(train_data, scaler_shl, scaler_as, scaler_std)
        val_data_norm = make_3channel_scaler_input(val_data, scaler_shl, scaler_as, scaler_std)
        scaler_str = 'AllSc'

    else:
        raise ValueError(f"Unknown scaler '{opt.scaler}'. Choose from: 'standard', 'powerlog', 'sinhlog', 'arcsinh', 'shiftedlog' ")
    
    # Post-Stretch standardization (skip for standard and allscalers)
    if opt.scaler not in ('standard', 'allscalers'):
        post_mean = train_data_norm.mean()
        post_std  = train_data_norm.std()
        train_data_norm = (train_data_norm - post_mean) / post_std
        val_data_norm   = (val_data_norm - post_mean) / post_std



    # --- Creating the Dataset for PyTorch ---
    # Create Datasets
    train_set = GalaxyDataset(train_data_norm, train_labels_int, augmentation=opt.augment, rotation=opt.rotation)
    val_set = GalaxyDataset(val_data_norm, val_labels_int, augmentation=False, rotation=opt.rotation)

# --- Suffix ---

rot_tag = ROT_TAG_MAP[opt.rotation]

suffix = _build_classification_suffix(model=opt.model, scaler=scaler_str, SB_limit=opt.SB_Limit, snapshots=opt.snapshots,n_classes=opt.n_classes, 
                                      augment=opt.augment, apply_exclusions=opt.exclude, online=opt.online_noise, split=opt.split_version, filter_str=opt.filer,
                                      rot_tag=rot_tag)



# --- Save scaler ---
if opt.online_noise:
    with open(f'{UZH_THESIS}/Scalers/scaler{suffix}.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler, 'omega': omega, 'sigma_sky': sigma_sky,
            'pixfovx': pixfovx, 'post_mean': post_mean, 'post_std': post_std, 'zp': 0,
        }, f)
elif opt.scaler == 'allscalers':
    with open(f'{UZH_THESIS}/Scalers/scalers{suffix}.pkl', 'wb') as f:
        pickle.dump({'shl': scaler_shl, 'as': scaler_as, 'std': scaler_std}, f)
else:
    with open(f'{UZH_THESIS}/Scalers/scaler{suffix}.pkl', 'wb') as f:
        pickle.dump(scaler, f)

# Create Dataloaders
batch_size = 32
n_workers = 1 if opt.online_noise else 0
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, 
                           num_workers=n_workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, 
                         num_workers=n_workers, pin_memory=True)

# --- Model ---
in_channels = 3 if opt.scaler == 'allscalers' else 1
if opt.model=='SimpleCNN':
    model = SimpleCNN(n_classes=opt.n_classes, in_channels=in_channels).to(device)          # Add different Models here later
elif opt.model=='CNN5_CLS':
    model = CNN5_CLS(n_classes=opt.n_classes).to(device)
else:
    raise ValueError(f'Unknown Model: {opt.model}')


# --- Training Setup ---
n_epochs = 100 if opt.augment else 50
patience = 10 if opt.augment else 6
lr = 1e-4
weight_decay= 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

patience_counter = 0
best_val_loss = float('inf')
losses = []
val_losses = []
for epoch in range(n_epochs):
    model.train()
    batch_loss = 0.0
    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        
    avg_loss = batch_loss / len(train_loader)
    losses.append(avg_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1}/{n_epochs}, LR: {current_lr:.2e}, Train Loss: {avg_loss:.4f}')
    
    # After training loop for this epoch:
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            val_loss += criterion(logits, y_batch).item()
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)
    print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {correct}/{len(val_set)}")

    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), f"{UZH_THESIS}/Model_Params/best_model{suffix}.pth")
        # print(f"New best model found at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
    else:
        patience_counter += 1
        # print(f"No improvement in val loss. Patience counter: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print("Training complete.")

np.save(f'{UZH_THESIS}/Losses/train_losses{suffix}.npy', np.array(losses))  
np.save(f'{UZH_THESIS}/Losses/val_losses{suffix}.npy', np.array(val_losses))
# np.save(f'{Thesis_path}/Losses/median_flux{suffix}.npy', np.array([median_flux]))
np.save(f'{UZH_THESIS}/Losses/sigma_sky{suffix}.npy', np.array([sigma_sky]))
if opt.scaler != 'standard':
    np.save(f'{UZH_THESIS}/Losses/post_stretch_stats{suffix}.npy', np.array([post_mean, post_std]))
    


