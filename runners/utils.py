import os
import torch
import torch.nn as nn
from PIL import Image
from datetime import datetime
from torchvision.utils import make_grid
from Register import Registers
import nibabel as nib


def remove_file(fpath):
    if os.path.exists(fpath):
        os.remove(fpath)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)
    return dir


def make_save_dirs(args, prefix, suffix=None, with_time=False):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if with_time else ""
    suffix = suffix if suffix is not None else ""

    result_path = make_dir(os.path.join(args.result_path, prefix, suffix, time_str))
    image_path = make_dir(os.path.join(result_path, "image"))
    log_path = make_dir(os.path.join(result_path, "log"))
    checkpoint_path = make_dir(os.path.join(result_path, "checkpoint"))
    sample_path = None
    sample_to_eval_path = None
    if args.sample_to_eval:
        sample_to_eval_path = make_dir(os.path.join(result_path, "sample_to_eval", os.path.basename(args.resume_model)))
    print("create output path " + result_path)
    return image_path, checkpoint_path, log_path, sample_path, sample_to_eval_path


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Parameter') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_optimizer(optim_config, parameters):
    if optim_config.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                                betas=(optim_config.beta1, 0.999))
    elif optim_config.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=optim_config.lr, momentum=0.9)
    else:
        return NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))


def get_dataset(data_config, test=False):
    if test:
        test_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage='test')
        return test_dataset
    train_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage='train')
    val_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage='valid')
    return train_dataset, val_dataset


@torch.no_grad()
def save_single_image(image, save_path, file_name, to_normal=True):
    image = image.detach().clone()
    if to_normal:
        image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image = image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    if image.shape[2] == 1:
        image = image[:,:,0]
    im = Image.fromarray(image)
    im.save(os.path.join(save_path, file_name))


@torch.no_grad()
def get_image_grid(batch, grid_size=4, to_normal=True):
    batch = batch.detach().clone()
    image_grid = make_grid(batch, nrow=grid_size)
    if to_normal:
        image_grid = image_grid.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image_grid = image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return image_grid


def save_syn_image(images_3d, raw_image_path, out_path, pid):
    raw_image = nib.load(raw_image_path)
    image = images_3d.permute(2, 1, 0).numpy()
    nii_img = type(raw_image)(image,
                              affine=raw_image.affine,
                              header=raw_image.header,
                              extra=raw_image.extra,
                              file_map=raw_image.file_map)
    nii_img.set_sform(raw_image.get_sform())
    nii_img.set_qform(raw_image.get_qform())
    nib.save(nii_img, out_path)
    
    print(f'saved_id: {pid}')
    
    return nii_img.get_fdata(), raw_image.get_fdata(), pid


def print_runtime(start_time, end_time):
    total_duration = end_time - start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    print(f"total time: {hours}h {minutes}m {seconds}s")

