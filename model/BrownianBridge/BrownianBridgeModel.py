import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model.utils import extract, default
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from model.BrownianBridge.bbdm_utils import file_path

class BrownianBridgeModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        # model hyperparameters
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        # UNet
        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        elif self.mt_type == 'control':
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
            m_t = np.sin(m_t * np.pi / 2)
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def input_condition_config(self, x, y, context):
        if self.condition_key == "nocond":
            context = None
        elif self.condition_key == 'hist_context':
            pass
        elif self.condition_key == "SpatialRescaler_context1":
            context = x[:,:1]
        elif self.condition_key == "hist_context_y_concat":
            context = {'concat': y, 'crossattn': context}
        else:
            context = y if context is None else context
        return x, y, context

    def forward(self, x, y, context=None):
        x, y, context = self.input_condition_config(x, y, context)

        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, context, t)

    def p_losses(self, x0, y, context, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        return recloss, log_dict

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_sub_batch(self, x_t, y, context, i, clip_denoised):
        HW = x_t.shape[-1]
        if HW > 128:
            sub_batch = 48 if HW >= 176 else 64
            full_batch = x_t.shape[0]
            num_iter = int(np.ceil(full_batch / sub_batch))
            x_t_recons = []
            x0_recons = []
            
            for n in range(num_iter):
                start_idx = n * sub_batch
                end_idx = min((n + 1) * sub_batch, full_batch)
                
                x_t_sub = x_t[start_idx:end_idx]
                y_sub = y[start_idx:end_idx]
                context_sub = None
                
                if self.condition_key == 'hist_context_y_concat':
                    context_sub = {}
                    for key in context.keys():
                        context_sub[key] = context[key][start_idx:end_idx]
                else:
                    context_sub = context[start_idx:end_idx]
                    
                x_t_recon, x0_recon = self.p_sample(x_t=x_t_sub, y=y_sub, context=context_sub, i=i, clip_denoised=clip_denoised)
                
                x_t_recons.append(x_t_recon)
                x0_recons.append(x0_recon)

            x_t = torch.cat(x_t_recons, dim=0)
            x0_recon = torch.cat(x0_recons, dim=0)
        else:
            x_t, x0_recon = self.p_sample(x_t=x_t, y=y, context=context, i=i, clip_denoised=clip_denoised)
        return x_t, x0_recon
    
    @torch.no_grad()
    def p_sample_loop(self, x, y, id=None, context=None, clip_denoised=True, sample_mid_step=False, path=None, save=False, config=None, device='cpu'):
        
        # save_img
        sample_step = self.model_config.BB.params.sample_step
        mid_slice = self.model_config.CondStageParams.in_channels // 2
        batch_size = y.shape[0]
        
        inference_type = config.model.BB.params.inference_type
        num_ISTA_step = config.model.BB.params.num_ISTA_step
        ISTA_step_size = config.model.BB.params.ISTA_step_size
        
        save_img = []
        if path is not None:
            sequence_path = os.path.join(path, 'sequence')
            if save and not os.path.exists(sequence_path):
                os.makedirs(os.path.join(path, 'sequence'), exist_ok=True)
        
        x, y, context = self.input_condition_config(x, y, context)

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample_sub_batch(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                if config.args.train:
                    img, x0_recon = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
                else:
                    img, x0_recon = self.p_sample_sub_batch(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
                
                if inference_type == 'normal' or config.args.train:
                    pass
                
                elif inference_type == 'average':
                    batch_size, ch_size, H, W = img.shape
                    radius = ch_size // 2
                    
                    # batch shape to averaged volume
                    padded_size = batch_size + (2 * radius)
                    averaged_volume = torch.zeros((ch_size, padded_size, H, W), device=device)
                    dup_slices = torch.ones(padded_size, dtype=torch.int32, device=device) * ch_size

                    for ch in range(ch_size):
                        averaged_volume[ch, ch:ch+batch_size] = img[:,ch]
                        dup_slices[ch] = ch + 1
                        dup_slices[-ch-1] = ch + 1

                    # B+2*radious, H, W
                    averaged_volume = torch.sum(averaged_volume, dim=0, keepdim=False) / dup_slices[:, None, None]

                    # averaged volume to batch shape
                    double_padded_size = batch_size + (4 * radius)
                    img = torch.zeros((ch_size, double_padded_size, H, W), device=device)
                    for ch in range(ch_size-1, -1, -1):
                        img[ch_size-1-ch, ch:ch+padded_size] = averaged_volume
                    
                    img = img[:, ch_size-1:ch_size-1+batch_size].permute(1, 0, 2, 3) # B, C, H, W
                    del averaged_volume, dup_slices
                    
                elif 'ISTA' in inference_type:
                    batch_size, ch_size, H, W = img.shape
                    radius = ch_size // 2
                    averaged_volume = self.batch2avgvolume(img, device, pad=True)
                    img = self.volume2batch(averaged_volume, img.shape, device)
                    
                    # correction
                    if i == len(self.steps) - 1:
                        continue
                    for iter in range(num_ISTA_step):
                        _, x0_recon = self.p_sample_sub_batch(x_t=img, y=y, context=context, i=i+1, clip_denoised=clip_denoised)
                        del _
                        score, var_t_nt = self.cal_score(x0=x0_recon, x_t=img, y=y, i=i+1)

                        # calulate step size
                        dim = torch.sqrt(torch.tensor(H * W, dtype=torch.float32, device=device))
                        if inference_type == 'ISTA_average':
                            score_volume = self.batch2avgvolume(score, device, pad=True)
                        elif inference_type == 'ISTA_mid':
                            score_volume = score[:, radius]
                            score_volume = torch.cat((score[0, :radius], score_volume, score[-1, radius+1:]), dim=0)
                            
                        score_l2_norm_squared = torch.sum(torch.pow(score_volume, 2), dim=(1, 2), keepdim=True)
                        gamma = ISTA_step_size * var_t_nt * (dim / score_l2_norm_squared)
                        # print(f"gamma: {gamma[55]}")
                        # print(f"gamma/ISTA_step_size: {gamma[55]/ISTA_step_size}")
                        # print(f"score_l2_norm: {torch.sqrt(score_l2_norm_squared[55])}")
                        # print(f"score_l2_norm_squared: {score_l2_norm_squared[55]}")
                        gamma_score_batch = self.volume2batch(gamma * score_volume, img.shape, device)
                            
                        img += gamma_score_batch
                    
                else:
                    raise NotImplementedError

            # append and save image
                if save:
                    save_img.append(img.detach().clone().cpu().numpy())
        
            if save and path is not None:
                for j in range(0, batch_size, 10):
                    saving_img = []
                    for k in range(0, len(save_img), 50):
                        saving_img.append(save_img[k][j, mid_slice])
                    saving_img.append(save_img[-1][j, mid_slice])
                    seq_img = np.concatenate(saving_img, axis=1)
                    pa_id = str(id[j], 'utf-8')
                    plt.imsave(file_path(pa_id, sequence_path), seq_img, cmap='gray')

            return img


    @torch.no_grad()
    def cal_score(self, x0, x_t, y, i):
        t = torch.full((x0.shape[0],), self.steps[i], device=x0.device, dtype=torch.long)
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        var_t_nt = extract(self.variance_t_tminus, t, x0.shape)

        score_t = - ((x_t - (1.-m_t)*x0 - m_t*y) / var_t)
        return score_t, var_t_nt[0]
        

    @torch.no_grad()
    def batch2avgvolume(self, batch_img, device, pad):
        batch_size, ch_size, H, W = batch_img.shape
        radius = ch_size // 2
        
        # batch shape to averaged volume
        padded_size = batch_size + (2 * radius)
        averaged_volume = torch.zeros((ch_size, padded_size, H, W), device=device)
        dup_slices = torch.ones(padded_size, dtype=torch.int32, device=device) * ch_size

        for ch in range(ch_size):
            averaged_volume[ch, ch:ch+batch_size] = batch_img[:,ch]
            dup_slices[ch] = ch + 1
            dup_slices[-ch-1] = ch + 1

        # B+2*radious, H, W
        averaged_volume = torch.sum(averaged_volume, dim=0, keepdim=False) / dup_slices[:, None, None]

        if not pad:
            averaged_volume = averaged_volume[radius:-radius]
        return averaged_volume

    @torch.no_grad()
    def volume2batch(self, volume_img, batch_img_shape, device):
        batch_size, ch_size, H, W = batch_img_shape
        radius = ch_size // 2
        padded_size = batch_size + (2 * radius)

        # averaged volume to batch shape
        double_padded_size = batch_size + (4 * radius)
        batch_img = torch.zeros((ch_size, double_padded_size, H, W), device=device)
        # batch_img = batch_img.to(device, non_blocking=True)
        for ch in range(ch_size-1, -1, -1):
            batch_img[ch_size-1-ch, ch:ch+padded_size] = volume_img
        
        batch_img = batch_img[:, ch_size-1:ch_size-1+batch_size].permute(1, 0, 2, 3) # B, C, H, W
        return batch_img
    
    @torch.no_grad()
    def sample(self, x, y, id=None, context=None, clip_denoised=True, sample_mid_step=False, path=None, save=False, config=None, device='cpu'):
        return self.p_sample_loop(x, y, id, context, clip_denoised, sample_mid_step, path, save, config, device)