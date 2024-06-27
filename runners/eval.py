import os
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def cal_mae(syn_img, raw_img):
    mae = np.abs(syn_img - raw_img).mean()
    return mae

def cal_mse(syn_img, raw_img):
    mse = ((syn_img - raw_img) ** 2).mean()
    return mse

def cal_rmse(syn_img, raw_img):
    rmse = np.sqrt(cal_mse(syn_img, raw_img))
    return rmse

def cal_nrmse(syn_img, raw_img):
    mse = cal_mse(syn_img, raw_img)
    data_range = np.max(syn_img) - np.min(syn_img)
    nrmse = np.sqrt(mse) / data_range
    return nrmse

def cal_psnr(syn_img, raw_img):
    psnr = peak_signal_noise_ratio(syn_img, raw_img, data_range=1.0)
    return psnr

def cal_ssim(syn_img, raw_img):
    ssim_index = structural_similarity(syn_img, raw_img, 
                                       data_range=1.0)
    return ssim_index

def calcul_metrics(metrics_dict, pa_id, syn_img, raw_img):
    metrics_dict[pa_id]['nrmse'] = cal_nrmse(syn_img, raw_img)
    metrics_dict[pa_id]['psnr'] = cal_psnr(syn_img, raw_img)
    metrics_dict[pa_id]['ssim'] = cal_ssim(syn_img, raw_img)
    print(f"{pa_id} : ")
    print(metrics_dict[pa_id])

def add_result(results_file, result_data):
    try:
        existing_data = pd.read_csv(results_file)
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=['name', 'date', 'size', 'baseline', 'plane', 'sampling', 'exp_type', 'nrmse', 'psnr', 'ssim'])

    result_data_columns = ['name', 'date', 'size', 'baseline', 'plane', 'sampling', 'exp_type', 'nrmse', 'psnr', 'ssim']
    new_row = pd.DataFrame([result_data], columns=result_data_columns)
    
    updated_data = pd.concat([existing_data, new_row], ignore_index=True)
    updated_data.to_csv(results_file, index=False)

def save_exp_result(results_file, config, means):
    try:
        existing_data = pd.read_csv(results_file)
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=['name', 'checkpoint', 'inference_type', 'nrmse', 'psnr', 'ssim'])
    
    name, checkpoint, inference_type = config.model.model_name, config.model.model_load_path, config.model.BB.params.inference_type
    checkpoint = checkpoint.split('/')[-1]
    
    result_data = [name, checkpoint, inference_type]
    for i in range(len(means)):
        result_data.append(means[i])
    
    result_data_columns = ['name', 'checkpoint', 'inference_type', 'nrmse', 'psnr', 'ssim']
    new_row = pd.DataFrame([result_data], columns=result_data_columns)
    
    updated_data = pd.concat([existing_data, new_row], ignore_index=True)
    updated_data.to_csv(results_file, index=False)
    
    