import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob
import gcsfs
from tqdm.notebook import tqdm
import torch


def make_dir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
        
def open_dataset(file_path):
    """Flexible opener that can handle both local files (legacy) and cloud urls. IMPORTANT: For this to work the `file_path` must be provided without extension."""
    if 'gs://' in file_path:
        store = f"{file_path}.zarr"
        ds = xr.open_dataset(store, engine='zarr')
    else:
        ds = xr.open_dataset(f"{file_path}.nc")
        # add information to sort and label etc
        ds.attrs['file_name']
    return ds
        
        
        
def prepare_predictor(data_sets, data_path,time_reindex=True):
    """
    Args:
        data_sets list(str): names of datasets
    """
        
    # Create training and testing arrays
    if isinstance(data_sets, str):
        data_sets = [data_sets]
        
    X_all      = []
    length_all = []
    
    for file in tqdm(data_sets):
        data = open_dataset(f"{data_path}inputs_{file}")
        X_all.append(data)
        length_all.append(len(data.time))
    
    X = xr.concat(X_all,dim='time')
    length_all = np.array(length_all)
    # X = xr.concat([xr.open_dataset(data_path + f"inputs_{file}.nc") for file in data_sets], dim='time')
    if time_reindex:
        X = X.assign_coords(time=np.arange(len(X.time)))

    return X, length_all

def prepare_predictand(data_sets,data_path,time_reindex=True):
    if isinstance(data_sets, str):
        data_sets = [data_sets]
        
    Y_all = []
    length_all = []
    
    for file in tqdm(data_sets):
        data = open_dataset(f"{data_path}outputs_{file}")
        Y_all.append(data)
        length_all.append(len(data.time))
    
    length_all = np.array(length_all)
    Y = xr.concat(Y_all,dim='time').mean('member')
    # Y = xr.concat([xr.open_dataset(data_path + f"outputs_{file}.nc") for file in data_sets], dim='time').mean("member")
    Y = Y.rename({'lon':'longitude','lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])
    if time_reindex:
        Y = Y.assign_coords(time=np.arange(len(Y.time)))
    
    return Y, length_all


def get_rmse(truth, pred):
    weights = np.cos(np.deg2rad(truth.latitude))
    return np.sqrt(((truth-pred)**2).weighted(weights).mean(['latitude', 'longitude'])).data.mean()

def plot_history(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()
    
# Utilities for normalizing the input data
def normalize(data, var, meanstd_dict):
    mean = meanstd_dict[var][0]
    std = meanstd_dict[var][1]
    return (data - mean)/std

def mean_std_plot(data,color,label,ax):
    
    mean = data.mean(['latitude','longitude'])
    std  = data.std(['latitude','longitude'])
    yr   = data.time.values

    ax.plot(yr,mean,color=color,label=label,linewidth=4)
    ax.fill_between(yr,mean+std,mean-std,facecolor=color,alpha=0.4)
    
    return yr, mean

def pytorch_train(model, optimizer, criterion, device, num_epochs, train_loader, val_loader):
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
       # training
       model.train()
       train_loss = 0.0
       for batch_X, batch_y in train_loader:
           batch_X = batch_X.to(device)
           batch_y = batch_y.to(device)
           # forward pass
           optimizer.zero_grad()
           outputs = model(batch_X)
           loss = criterion(outputs, batch_y)
           # backward pass
           loss.backward()
           optimizer.step()
           train_loss += loss.item()
    
        # validation
       model.eval()
       val_loss = 0.0
       with torch.no_grad():
           for batch_X, batch_y in val_loader:
               batch_X = batch_X.to(device)
               batch_y = batch_y.to(device)
               
               outputs = model(batch_X)
               loss = criterion(outputs, batch_y)
               val_loss += loss.item()
       
       train_loss /= len(train_loader)
       val_loss /= len(val_loader)
    
       train_losses.append(train_loss)
       val_losses.append(val_loss)
       
       print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
       
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           patience_counter = 0
       else:
           patience_counter += 1
           if patience_counter >= patience:
               print(f'Early stopping at epoch {epoch+1}')
               break
   
    return train_losses, val_losses

def pytorch_train_VAE(model, optimizer, criterion, device, num_epochs, train_loader, val_loader, patience=20):    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_size = batch_X.size(0)
            
            optimizer.zero_grad()
            recon, z_mean, z_log_var = model(batch_y, batch_X)
            loss = criterion(batch_y, recon, z_mean, z_log_var)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_size
            total_train_samples += batch_size
        
        train_loss /= total_train_samples
        
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_size = batch_X.size(0)
                
                recon, z_mean, z_log_var = model(batch_y, batch_X) 
                loss = criterion(batch_y, recon, z_mean, z_log_var)
                
                val_loss += loss.item() * batch_size
                total_val_samples += batch_size
        
        val_loss /= total_val_samples
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}.')
                break
    
    return train_losses, val_losses

def prepare_spatial_data(X_train, y_train, lat, lon, n_subset=15000):
    """
    Flatten temporal-spatial data and add lat/lon as features.
    
    Args:
        X_train: (n_time, 2) - CO2, CH4
        y_train: (n_time, n_space) - temperature at each grid point
        lat: (n_lat,) latitude values
        lon: (n_lon,) longitude values
        n_subset: number of random samples for training
    
    Returns:
        X_flat: (n_subset, 4) - [CO2, CH4, lat, lon]
        y_flat: (n_subset,)
    """
    n_time, n_space = y_train.shape
    n_lat, n_lon = len(lat), len(lon)
    
    # Create meshgrid of lat/lon
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    lat_flat = lat_grid.flatten()  # (n_space,)
    lon_flat = lon_grid.flatten()  # (n_space,)
    
    # Random subset of (time, space) indices
    np.random.seed(42)
    idx = np.random.choice(n_time * n_space, n_subset, replace=False)
    time_idx, space_idx = np.unravel_index(idx, (n_time, n_space))
    
    # Build feature matrix: [CO2, CH4, lat, lon]
    X_flat = np.column_stack([
        X_train[time_idx, 0],      # CO2
        X_train[time_idx, 1],      # CH4
        lat_flat[space_idx],       # lat
        lon_flat[space_idx]        # lon
    ])
    y_flat = y_train[time_idx, space_idx]
    
    return X_flat, y_flat, lat_flat, lon_flat


def prepare_test_spatial(X_test, lat_flat, lon_flat):
    """
    Create test features for all (time, space) combinations.
    
    Args:
        X_test: (n_test_time, 2)
        lat_flat, lon_flat: (n_space,) each
    
    Returns:
        X_test_full: (n_test_time * n_space, 4)
    """
    n_test = X_test.shape[0]
    n_space = len(lat_flat)
    
    # Repeat each test time point for all spatial locations
    X_test_full = np.column_stack([
        np.repeat(X_test[:, 0], n_space),  # CO2
        np.repeat(X_test[:, 1], n_space),  # CH4
        np.tile(lat_flat, n_test),          # lat
        np.tile(lon_flat, n_test)           # lon
    ])
    return X_test_full

    