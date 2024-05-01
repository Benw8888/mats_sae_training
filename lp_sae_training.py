import os
import sys

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

cfg = LanguageModelSAERunnerConfig(

    # Data Generating Function (Model + Training Distibuion)
    model_name = "EleutherAI/pythia-14m", # "gpt2-small", #"EleutherAI/pythia-14m", #"gpt2-small",
    hook_point = "blocks.{layer}.hook_resid_pre",
    hook_point_layer = 3, # 6
    d_in = 128, # 768
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,
    
    # SAE Parameters
    expansion_factor = 64, #[16,32,64],
    b_dec_init_method = "geometric_median",
    
    # Training Parameters
    lr = 4e-4, #5e-5, #4e-4
    l1_coefficient = 
        # 8e-9,
        # 8e-8,
        # 8e-7, 
        # 8e-6, # goal: a bit less than this
        # # 4e-6,
        # # 1.8e-5,
        # 8e-5,
        
        # # gpt2 L0.6
        # 2.5e-8,
        # 5e-8,
        # 1e-7, 
        # 2e-7,
        # 4e-7,
        # 8e-7,
        # 1.6e-6,
        
        # gpt2 L0.6 extension
        # 1.25e-8,
        # 1.77e-8,
        # 3.54e-8,
        # 7e-8,
        # 1.4e-7,
        # 2.8e-7,
        # 5.7e-7,
        # 11.3e-7, 
        
        # # gpt2 L0.6 extension 2
        # 2.26e-6,
        # 3.2e-6,
        # 4.53e-6,
        # 6.4e-6,
               
        # # gpt2 L1 exp 16
        # 1e-5,
        # 2e-5,
        # 4e-5,
        # 8e-5,
        
        # # gpt2 L1 exp 16 extension
        # 5e-6,
        # 0.7e-5,
        # 1.4e-5,
        # 2.8e-5,
        # 5.7e-5,
        
        # # gpt2 L1 extension 2
        # 1.13e-4,
        # 1.6e-4,
        # 2.26e-4,
        
        # # gpt2 lp^p
        # 3e-5,
        # 5e-5,
        # 7e-5,
        # 1e-4,
        # 1.4e-4,
        
        
        # # pythia 14m L0.6,L0.8 extension (needs to be run twice)
        # 1e-9,
        # 1e-10,
        # 1e-11,
        # 1e-12,
        # 1e-13,
        # 0,
        # 1e-3,

        # # pythia 14m L0.6, 0.8
        # 3e-9,   
        # 5.6e-9,
        # 1e-8,
        # 1.8e-8,
        # 3e-8,
        # 5.6e-8,
        # 1e-7,
        # 1.8e-7,
        # 3e-7,
        # 5.6e-7,
        # 1e-6,
        # 1.8e-6,
        # 3e-6,
        # 5.6e-6,
        # 1e-5,
        # 1.8e-5,
        # 3e-5,
        # 5.6e-5,
        # 1e-4,
        # 1.8e-4,
        # 3e-4,
        
        # # pythia 14m L1
        # 1e-4,    
        # 1.33e-4,
        # 1.8e-4,
        # 2.37e-4,
        # 3e-4,
        # 4.22e-4,
        # 5.62e-4,
        # 7.50e-4,
        # 1e-3,
        # 1.33e-3,
        # 1.8e-3,
        # 2.37e-3,
        # 3e-3,
        
        # # Pythia 14m L1 extension (needs to be run twice)
        # 1e-5,
        # 1e-6,
        # 1e-7,
        # 1e-8,
        # 1e-9,
        # 0,
        
        # pythia 14m lp^p
        # 1e-8,
        # 1e-7,
        # 1e-6,
        # 3e-6,
        # 1e-5,
        # 1.8e-5,
        # 3e-5,
        # 5.6e-5,
        # 1e-4,    
        # 1.8e-4,
        # 3e-4,
        # 4.22e-4,
        # 5.62e-4,
        # 7.50e-4,
        # 1e-3,
        # 1.33e-3,
        # 1.8e-3,
        # 2.37e-3,
        # 3e-3,
        # 1e-2,
        
        # testing p annealing, pythia
        1.33e-3
    ,
    lp_norm = 
        1
    ,
    lr_scheduler_name="constantwithwarmup",
    train_batch_size = 4096,
    context_size = 128,
    lr_warm_up_steps=5000,
    anneal = True,
    
    # Activation Store Parameters
    n_batches_in_buffer = 128,
    total_training_tokens = 300_000_000,
    store_batch_size = 32,
    
    # Dead Neurons and Sparsity
    use_ghost_grads=False,
    feature_sampling_window = 1000,
    dead_feature_window=5000,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "mats_sae_training_gpt2",
    wandb_entity = None,
    wandb_log_frequency=100,
    
    # Misc
    device = "cuda",
    seed = 40, #42
    n_checkpoints = 3,
    checkpoint_path = "checkpoints",
    dtype = torch.float32,
    use_cached_activations = False,
)

sparse_autoencoder = language_model_sae_runner(cfg)