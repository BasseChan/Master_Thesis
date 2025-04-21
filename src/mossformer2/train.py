import os, random
import numpy as np
import torch
from types import SimpleNamespace

from dataloader.dataloader import get_dataloader
from solver import Solver
from networks import network_wrapper

def train_model(args):
    # Ustawienie seedów i środowiska
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTORCH_SEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    args.device = device

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', rank=args.local_rank,
            init_method='env://', world_size=args.world_size
        )

    # Ładowanie modelu
    model = network_wrapper(args).ss_network.to(device)

    if not args.distributed or args.local_rank == 0:
        print(f"Started training, checkpoint dir: {args.checkpoint_dir}")
        print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optymalizator
    if args.network in ['MossFormer2_SS_16K','MossFormer2_SS_8K']:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_learning_rate)
    else:
        raise NotImplementedError(f"Network {args.network} is not implemented.")

    # Dataloadery
    train_sampler, train_generator = get_dataloader(args, 'train')
    _, val_generator = get_dataloader(args, 'val')
    _, test_generator = get_dataloader(args, 'test') if args.tt_list else (None, None)
    args.train_sampler = train_sampler

    # Inicjalizacja solvera
    solver = Solver(
        args=args,
        model=model,
        optimizer=optimizer,
        train_data=train_generator,
        validation_data=val_generator,
        test_data=test_generator
    )

    # Trenowanie
    solver.train()

# PRZYKŁADOWE ARGUMENTY – dostosuj do swoich danych / ścieżek
args_dict = {
    'seed': 42,
    'use_cuda': torch.cuda.is_available(),
    'distributed': False,
    'local_rank': 0,
    'world_size': 1,

    'checkpoint_dir': 'checkpoints/trained',
    'network': 'MossFormer2_SS_16K',
    'train_from_last_checkpoint': 0,
    'init_checkpoint_path': None,

    'batch_size': 8,
    'init_learning_rate': 0.001,
    'max_epoch': 5,
    'print_freq': 10,
    'checkpoint_save_freq': 50,
    
    'tr_list': 'path/to/train_list.txt',
    'cv_list': 'path/to/val_list.txt',
    'tt_list': None,
    'load_type': 'one_input_one_output',
    'accu_grad': 1,
    'max_length': 16000,
    'num_workers': 4,
    'sampling_rate': 16000,
    'load_fbank': 1,
    
    'num_spks': 2,
    'encoder_kernel_size': 16,
    'encoder_embedding_dim': 512,
    'mossformer_sequence_dim': 512,
    'num_mossformer_layer': 24,

    'effec_batch_size': 8,
    'num_gpu': 1,
    'finetune_learning_rate': 0.0001,
    'weight_decay': 1e-5,
    'clip_grad_norm': 10.0,
    'loss_threshold': -9999.0,

    'train_from_last_checkpoint': 1,
    'init_checkpoint_path': 'mossformer2/checkpoints',
}

# Konwersja na obiekt z atrybutami
train_args = SimpleNamespace(**args_dict)

