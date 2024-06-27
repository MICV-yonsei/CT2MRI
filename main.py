import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np

from utils import dict2namespace, get_runner, namespace2dict
import torch.multiprocessing as mp
import torch.distributed as dist

import wandb


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('-c', '--config', type=str, default='BB_base.yml', help='Path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results', help="The directory to save results")

    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--sample_to_eval', action='store_true', default=False, help='sample for evaluation')
    parser.add_argument('--sample_at_start', action='store_true', default=False, help='sample at start(for debug)')
    parser.add_argument('--save_top', action='store_true', default=False, help="save top loss checkpoint")

    # system
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')
    parser.add_argument('--port', type=str, default='12355', help='DDP master port')
    # resume
    parser.add_argument('--resume_model', type=str, default=None, help='model checkpoint')
    parser.add_argument('--resume_optim', type=str, default=None, help='optimizer checkpoint')

    # name
    parser.add_argument('--exp_name', type=str, help='experiment name (result dir name)')
    # training
    parser.add_argument('--max_epoch', type=int, default=None, help='optimizer checkpoint')
    parser.add_argument('--max_steps', type=int, default=None, help='optimizer checkpoint')
    # data
    parser.add_argument('--dataset_type', type=str, default='' ,help='dataset type (search register)')
    parser.add_argument('--plane', type=str, help='input view: axial, sagittal, coronal')
    parser.add_argument('--HW', type=int, default=128, help='HW of input image')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    # model
    parser.add_argument('--mt_type', type=str, default=None, help='mt_type')
    parser.add_argument('--objective', type=str, default=None, help='objective')
    parser.add_argument('--loss_type', type=str, default=None, help='loss_type')
    parser.add_argument('--sample_step', type=int, default=None, help='sample step')
    parser.add_argument('--num_timesteps', type=int, default=None, help='num_timesteps')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='ddim_eta')
    parser.add_argument('--max_var', type=float, default=1.0, help='max_var s')
    parser.add_argument('--inference_type', type=str, default=None, help='inference_type: normal, average, ISTA_average, ISTA_mid')
    parser.add_argument('--ISTA_step_size', type=float, default=None, help='ISTA_step_size')
    parser.add_argument('--num_ISTA_step', type=int, default=None, help='num_ISTA_step')
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    namespace_config.data.dataset_name += f"_{args.HW}"
    namespace_config.data.dataset_config.image_size = args.HW
    namespace_config.data.train.batch_size = args.batch
    namespace_config.data.val.batch_size = args.batch
    namespace_config.data.test.batch_size = args.batch
    
    namespace_config.model.model_name = args.exp_name
    namespace_config.model.BB.params.UNetParams.image_size = args.HW
    namespace_config.model.BB.params.eta = args.ddim_eta
    namespace_config.model.BB.params.max_var = args.max_var
    
    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch
    if args.max_steps is not None:
        namespace_config.training.n_steps = args.max_steps
    # model
    if args.mt_type is not None:
        namespace_config.model.BB.params.mt_type = args.mt_type
    if args.objective is not None:
        namespace_config.model.BB.params.objective = args.objective
    if args.loss_type is not None:
        namespace_config.model.BB.params.loss_type = args.loss_type
    if args.sample_step is not None:
        namespace_config.model.BB.params.sample_step = args.sample_step
    if args.num_timesteps is not None:
        namespace_config.model.BB.params.num_timesteps = args.num_timesteps
    if args.inference_type is not None:
        namespace_config.model.BB.params.inference_type = args.inference_type
    if args.ISTA_step_size is not None:
        namespace_config.model.BB.params.ISTA_step_size = args.ISTA_step_size
    if args.num_ISTA_step is not None:
        namespace_config.model.BB.params.num_ISTA_step = args.num_ISTA_step
        
    if args.sample_to_eval:
        if args.dataset_type != '':
            namespace_config.data.dataset_type = args.dataset_type

    dict_config = namespace2dict(namespace_config)
    
    config_dict = {"train": args.train,
                   "gpu_ids": args.gpu_ids,
                   "dataset_type": namespace_config.data.dataset_type,
                   "image_size": namespace_config.data.dataset_config.image_size,
                   "max_epoch": namespace_config.training.n_epochs,
                   "n_steps": namespace_config.training.n_steps,
                   "batch_size": namespace_config.data.train.batch_size,
                   "mt_type": namespace_config.model.BB.params.mt_type,
                   "objective": namespace_config.model.BB.params.objective,
                   "loss_type": namespace_config.model.BB.params.loss_type,
                   "sample_step": namespace_config.model.BB.params.sample_step,
                   "num_timesteps": namespace_config.model.BB.params.num_timesteps,
                   "ddim_eta": namespace_config.model.BB.params.eta,
                   "max_var": namespace_config.model.BB.params.max_var,
                   }
    if not args.sample_to_eval:
        try:
            wandb.init(project="", entity="", name=namespace_config.model.model_name, config=config_dict)
        except:
            print('Could not init wandb')
    
    return namespace_config, dict_config


def set_random_seed(SEED=1234):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def DDP_run_fn(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.args.port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    set_random_seed(config.args.seed)

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    config.training.device = [torch.device("cuda:%d" % local_rank)]
    print('using device:', config.training.device)
    config.training.local_rank = local_rank
    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


def CPU_singleGPU_launcher(config):
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


def DDP_launcher(world_size, run_fn, config):
    mp.spawn(run_fn,
             args=(world_size, copy.deepcopy(config)),
             nprocs=world_size,
             join=True)


def main():
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args

    gpu_ids = args.gpu_ids
    if gpu_ids == "-1": # Use CPU
        nconfig.training.use_DDP = False
        nconfig.training.device = [torch.device("cpu")]
        CPU_singleGPU_launcher(nconfig)
    else:
        gpu_list = gpu_ids.split(",")
        if len(gpu_list) > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            nconfig.training.use_DDP = True
            DDP_launcher(world_size=len(gpu_list), run_fn=DDP_run_fn, config=nconfig)
        else:
            nconfig.training.use_DDP = False
            nconfig.training.device = [torch.device(f"cuda:{gpu_list[0]}")]
            CPU_singleGPU_launcher(nconfig)
    return


if __name__ == "__main__":
    main()
