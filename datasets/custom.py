import torch
from torch.utils.data import Dataset

from Register import Registers
from datasets.base import multi_ch_nifti_default_Dataset
import os
import numpy as np
import h5py
import pickle

@Registers.datasets.register_with_name('BraTS_t2f_t1n_aligned_global_hist_context')
class hist_context_BraTS_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        
        hdf5_path = os.path.join(dataset_config.dataset_path, f"BraTS_t2f_to_t1n_{stage}_{dataset_config.plane}.hdf5")
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('target_dataset'))
            B_dataset = np.array(hf.get('source_dataset'))
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))

        hist_type = dataset_config.hist_type
        hist_path = os.path.join(dataset_config.dataset_path, f"MR_hist_global_t1n_{stage}_{dataset_config.plane}_BraTS_.pkl")
        if stage == 'test' and hist_type is not None:
            hist_path = os.path.join(dataset_config.dataset_path, f"MR_hist_global_t1n_{stage}_{dataset_config.plane}_BraTS_"+hist_type+".pkl")   
        print(hist_path)
        with open(hist_path, 'rb') as f:
            self.hist_dict = pickle.load(f)

        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(A_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = multi_ch_nifti_default_Dataset(B_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        
    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        out_ori = self.imgs_ori[i] # (3, 160, 160)
        out_cond = self.imgs_cond[i] # (3, 160, 160)
        out_hist = self.hist_dict[out_cond[1].decode('utf-8')]
        out_hist = torch.from_numpy(out_hist).float() # (32, 128, 1)

        return out_ori, out_cond, out_hist
    
@Registers.datasets.register_with_name('BraTS_t2f_t1n_aligned')
class BraTS_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        
        hdf5_path = os.path.join(dataset_config.dataset_path, f"BraTS_t2f_to_t1n_{stage}_{dataset_config.plane}.hdf5")
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('target_dataset'))
            B_dataset = np.array(hf.get('source_dataset'))
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))

        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(A_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = multi_ch_nifti_default_Dataset(B_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]

@Registers.datasets.register_with_name('ct2mr_aligned_global_hist_context')
class hist_context_CT2MR_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        
        hdf5_path = os.path.join(dataset_config.dataset_path, f"{dataset_config.image_size}_{stage}_{dataset_config.plane}.hdf5")
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('MR_dataset'))
            B_dataset = np.array(hf.get('CT_dataset'))
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))
        
        hist_type = dataset_config.hist_type
        hist_path = os.path.join(dataset_config.dataset_path, f"MR_hist_global_{dataset_config.image_size}_{stage}_{dataset_config.plane}_.pkl")                
        if stage == 'test' and hist_type is not None:
            hist_path = os.path.join(dataset_config.dataset_path, f"MR_hist_global_{dataset_config.image_size}_{stage}_{dataset_config.plane}_"+hist_type+".pkl")                
        print(hist_path)
        with open(hist_path, 'rb') as f:
            self.hist_dict = pickle.load(f)

        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(A_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = multi_ch_nifti_default_Dataset(B_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        
    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        out_ori = self.imgs_ori[i] # (3, 160, 160)
        out_cond = self.imgs_cond[i] # (3, 160, 160)
        out_hist = self.hist_dict[out_cond[1].decode('utf-8')]
        out_hist = torch.from_numpy(out_hist).float() # (32, 128, 1)

        return out_ori, out_cond, out_hist

    
@Registers.datasets.register_with_name('ct2mr_aligned')
class CT2MR_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        
        hdf5_path = os.path.join(dataset_config.dataset_path, f"{dataset_config.image_size}_{stage}_{dataset_config.plane}.hdf5")
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('MR_dataset'))
            B_dataset = np.array(hf.get('CT_dataset'))
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))
            
        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(A_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = multi_ch_nifti_default_Dataset(B_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]

