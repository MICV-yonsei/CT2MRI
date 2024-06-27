import os
import numpy as np
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from PIL import Image
from tqdm.autonotebook import tqdm
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import *
from runners.eval import calcul_metrics, save_exp_result

import nibabel as nib
from collections import defaultdict
import pandas as pd

import time
import wandb

@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        # elif config.model.model_type == "LBBDM":
        #     bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_name), (x_cond, x_cond_name), *context= batch
            context = context[0] if context else None

            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_name), (x_cond, x_cond_name), *context= batch
            context = context[0] if context else None
            
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        print(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        print(self.net.ori_latent_mean)
        print(self.net.ori_latent_std)
        print(self.net.cond_latent_mean)
        print(self.net.cond_latent_std)

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name), (x_cond, x_cond_name), *context = batch
        context = context[0] if context else None

        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])
        if context is not None:
            context = context.to(self.config.training.device[0])

        loss, additional_info = net(x, x_cond, context=context)
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            try:
                wandb.log({f"loss/{stage}": loss}, step=step)
            except:
                print(f'Could not log loss to wandb')
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
                try:
                    wandb.log({f"recloss_noise/{stage}": additional_info["recloss_noise"]}, step=step)
                except:
                    print(f'Could not log recloss_noise to wandb')   
                        
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
                try:
                    wandb.log({f"recloss_xy/{stage}": additional_info["recloss_xy"]}, step=step)
                except:
                    print(f'Could not log recloss_xy to wandb')
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))

        (x, x_name), (x_cond, x_cond_name), *context = batch
        context = context[0] if context else None

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])
        if context is not None:
            context = context[0:batch_size].to(self.config.training.device[0])

        grid_size = 4
        sample = net.sample(x, x_cond, context=context, clip_denoised=self.config.testing.clip_denoised, config=self.config, device=self.config.training.device[0]).to('cpu')
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        mid_slice_index = image_grid.shape[-1] // 2
        image_grid = image_grid[:,:,mid_slice_index:mid_slice_index+1]
        im = Image.fromarray(image_grid[:,:,0])
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')
            try:
                wandb.log({f'{stage}_skip_sample': [wandb.Image(image_grid, caption=f'{stage}_skip_sample')]}, step=self.global_step)
            except:
                print(f'Could not log {stage}_skip_sample to wandb')
            
        image_grid = get_image_grid(x_cond.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        image_grid = image_grid[:,:,mid_slice_index:mid_slice_index+1]
        im = Image.fromarray(image_grid[:,:,0])
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')
            try:
                wandb.log({f'{stage}_condition': [wandb.Image(image_grid, caption=f'{stage}_condition')]}, step=self.global_step)
            except:
                print(f'Could not log {stage}_condition to wandb')
                
        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        image_grid = image_grid[:,:,mid_slice_index:mid_slice_index+1]
        im = Image.fromarray(image_grid[:,:,0])
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')
            try:
                wandb.log({f'{stage}_ground_truth': [wandb.Image(image_grid, caption=f'{stage}_ground_truth')]}, step=self.global_step)
            except:
                print(f'Could not log {stage}_ground_truth to wandb')

    @torch.no_grad()
    def sample_to_eval(self, net, test_dataset, sample_path):
        start_time = time.time()
        
        raw_data_dir = "/root_dir/datasets/raw_data_tight"
        if 'BraTS' in sample_path:
            raw_data_dir = "/root_dir/BraTS/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"

        mid_slice = self.config.data.dataset_config.channels // 2
        H = self.config.data.dataset_config.image_size
        sample_step = self.config.model.BB.params.sample_step
        inference_type = self.config.model.BB.params.inference_type
        num_ISTA_step = self.config.model.BB.params.num_ISTA_step
        ISTA_step_size = self.config.model.BB.params.ISTA_step_size
        dataset_type = self.config.data.dataset_type

        sample_path = os.path.join(sample_path, f"{inference_type}_{sample_step}")
        if 'ISTA' in inference_type:
            sample_path = os.path.join(sample_path, f"{inference_type}_{sample_step}_{ISTA_step_size}_{num_ISTA_step}")
            
        if "colin" in dataset_type:
            sample_path += '_colin'
        elif "best" in dataset_type:
            sample_path += '_best_meanmax'
        elif "average" in dataset_type:
            sample_path += '_average'
            
        print(f"sample_path: {sample_path}")
        os.makedirs(sample_path, exist_ok=True)
        
        batch_dict = defaultdict(list)
        for idx in range(len(test_dataset)):
            pid = test_dataset[idx][0][1].decode('utf-8')
            batch_dict[pid].append(test_dataset[idx])
        
        metrics_dict = defaultdict(dict)
        for pid in tqdm(batch_dict.keys()):
            out_path = os.path.join(sample_path, f'{pid}.nii')
            if os.path.exists(out_path):
                syn_img = nib.load(out_path).get_fdata()
                syn_img = np.nan_to_num(syn_img)
                raw_img = nib.load(os.path.join(raw_data_dir, pid, f"cropped_{pid}-t1n_preprocessed.nii")).get_fdata()
                raw_img = np.nan_to_num(raw_img)
                calcul_metrics(metrics_dict, pid, syn_img, raw_img)
                continue

            test_batch = default_collate(batch_dict[pid])
            (x, x_name), (x_cond, x_cond_name), *context = test_batch
            context = context[0] if context else None
            x_cond = x_cond.to(self.config.training.device[0], non_blocking=True)            
            if context is not None:
                context = context.to(self.config.training.device[0], non_blocking=True)

            sample = net.sample(x, x_cond, x_cond_name, context=context, clip_denoised=False, path=sample_path, save=False, config=self.config, device=self.config.training.device[0])
            sample = sample[:, mid_slice].detach().clone().cpu().mul_(0.5).add_(0.5).clamp_(0, 1.)
            
            raw_image_path = os.path.join(raw_data_dir, pid, f"cropped_MR_preprocessed_{H}.nii")
            if 'BraTS' in sample_path:
                raw_image_path = os.path.join(raw_data_dir, pid, f"cropped_{pid}-t1n_preprocessed.nii")
            syn_img, raw_img, pid = save_syn_image(sample, raw_image_path, out_path, pid)
            calcul_metrics(metrics_dict, pid, syn_img, raw_img)
                    
        df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        means = df.mean()
        df.loc['mean'] = means
                
        df.to_csv(os.path.join(sample_path, 'results.csv'), index_label='pa_id')
        
        results_file = os.path.join('/root_dir/code/results/test_results.csv')
        save_exp_result(results_file, self.config, means)

        end_time = time.time()
        print_runtime(start_time, end_time)
        