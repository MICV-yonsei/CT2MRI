# Slice-Consistent 3D Volumetric Brain CT-to-MRI Translation with 2D Brownian Bridge Diffusion Model

**Early accepted at MICCAI 2024**

[[Project Page]](https://micv-yonsei.github.io/ct2mri2024/) [[paper]](https://arxiv.org/pdf/2407.05059) [[arXiv]](https://arxiv.org/pdf/2407.05059) 

### Requirements

```
conda env create -f environment.yml
conda activate ct2mri
```


### Dataset Preparation

For datasets from BraTS2023 that include paired multi-modal MRI images, your dataset directory should be structured as follows:
- training set: /root_dir/BraTS/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
- validation set: /root_dir/BraTS/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData

To create an HDF5 file for efficient data loading, run the following command:
```commandline
sh shell/data/make_hdf5_BraTS.sh
```

To generate a histogram dataset (in .pkl format) for Style Key Conditioning (SKC), execute the command below:
```commandline
sh shell/data/make_hist_dataset_BraTS.sh
```

For custom CT-MR datasets, ensure to modify the `data_dir` and `data_csv` arguments in the `make_hdf5.sh` script to match your custom dataset paths:
```commandline
sh shell/data/make_hdf5.sh
```

To generate a histogram dataset (in .pkl format) for Style Key Conditioning (SKC) with a custom CT-MR dataset, modify the `data_dir` and `data_csv` arguments in the `make_hist_dataset.sh` script to match your custom dataset paths:
```commandline
sh shell/data/make_hist_dataset.sh
```


### Training

For training with the BraTS dataset:
```commandline
sh shell/train/train_BraTS.sh
```

For training with a custom CT-MR dataset, use the following command:
```commandline
sh shell/train/train.sh
```


### Testing

For testing with the BraTS dataset:
```commandline
sh shell/train/test_BraTS.sh
```

For testing with a custom CT-MR dataset, use the following command:
```commandline
sh shell/train/test.sh
```


## Acknowledgement

Our code was implemented based on the code from [BBDM](https://github.com/xuekt98/BBDM). We are grateful to Bo Li, Kai-Tao Xue, et al.
