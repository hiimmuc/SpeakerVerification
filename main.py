import argparse
from importlib_metadata import metadata

import torch.distributed as dist
import torch.multiprocessing as mp
from export import *
from inference import inference
from dataprep import DataGenerator
from trainer import train
from utils import read_config
import subprocess

def main(args):
    if args.do_train:
        # TODO: train model
        try:
            if args.distributed:
                npugs = torch.cuda.device_count()
                mp.spawn(train, nprocs=npugs, args=(npugs, args))
            else:
                train(0, None, args)
        except:
            train(0, None, args)
            
    elif args.do_infer:
        # TODO: evaluate model
        inference(args)
        
    elif args.do_export:
        export_model(args, check=True)
    else:
        raise 'Wrong main mode, available: do_train, do_infer, do_export'
    pass


#--------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description="SpeakerNet")
if __name__ == '__main__':
    # YAML
    parser.add_argument('--config', type=str, default=None)

    # control flow
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_infer', action='store_true', default=False)
    parser.add_argument('--do_export', action='store_true', default=False)
    
    # Infer mode
    parser.add_argument('--eval',
                        dest='eval',
                        action='store_true',
                        help='Eval only')
    parser.add_argument('--test',
                        dest='test',
                        action='store_true',
                        help='Test only')
    parser.add_argument('--predict',
                        dest='predict',
                        action='store_true',
                        help='Predict')
    
    # Device settings
    parser.add_argument('--device',
                        type=str,
                        default="cuda",
                        help='cuda or cpu')
    parser.add_argument('--distributed', 
                        action='store_true', 
                        default=False, 
                        help='Decide whether to use multi gpus')
    parser.add_argument('--distributed_backend',
                        type=str,
                        default="nccl",
                        help='nccl or gloo or mpi')
    
    ## Distributed and mixed precision training
    parser.add_argument('--port',           
                        type=str,   
                        default="8888", 
                        help='Port for distributed training, input as text')
    parser.add_argument('--mixedprec',      
                        dest='mixedprec',   
                        action='store_true', 
                        default=False,
                        help='Enable mixed precision training')


    parser.add_argument('--augment',
                        action='store_true',
                        default=False,
                        help='Augment input')
    
    parser.add_argument('--early_stopping',
                        action='store_true',
                        default=False,
                        help='Early stopping')
    
    parser.add_argument('--seed',
                        type=int,
                        default=1000,
                        help='seed')
    
   #--------------------------------------------------------------------------------------#

    sys_args = parser.parse_args()

    if sys_args.config is not None:
        args = read_config(sys_args.config, sys_args)
        args = argparse.Namespace(**args)
    
    # Initialise directories
    model_save_path = os.path.join(
        args.save_folder, Path(f"{args.model['name']}/{args.criterion['name']}/model"))
    os.makedirs(model_save_path, exist_ok=True)
    
    result_save_path = os.path.join(
        args.save_folder, Path(f"{args.model['name']}/{args.criterion['name']}/result"))
    os.makedirs(result_save_path, exist_ok=True)
    
    config_clone_path = os.path.join(
        args.save_folder, Path(f"{args.model['name']}/{args.criterion['name']}/config"))
    
    if not os.path.exists(config_clone_path):
        os.makedirs(config_clone_path, exist_ok=True)
        if args.config is not None:
            config_dir = '/'.join(str(args.config).split('/')[:-1])
            subprocess.call(f"cp -R {config_dir}/*.yaml {config_clone_path}", shell=True)
    
    # parse metadata to save files
    metadata_path = os.path.join(args.save_folder, 'metadata')
    if os.path.exists(os.path.join(args.data_folder, 'metadata')) and os.path.exists(metadata_path):
        print("Metadata files are exist, skip preparing...")
    else:
        print("Metadata files are not exist, start preparing...")
        os.makedirs(metadata_path, exist_ok=True)
        data_generator = DataGenerator(args)        
        valid_spks, invalid_spks = data_generator.generate_metadata()
        args.nClasses = len(valid_spks)
    
    # Run
    n_gpus = torch.cuda.device_count()
    
    print('Seed:', args.seed)
    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    
    main(args)
    
    #######################################################################################