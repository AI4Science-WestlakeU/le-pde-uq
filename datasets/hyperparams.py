INVALID_VALUE = -200

def update_legacy_default_hyperparam(Dict):
    """Default hyperparameters for legacy settings."""
    default_param = {
        # Dataset:
        "time_interval": 1,
        "sector_size": "-1",
        "sector_stride": "-1",
        "seed": -1,
        "dataset_split_type": "standard",
        "train_fraction": float(8/9),
        "temporal_bundle_steps": 1,
        "is_y_variable_length": False,
        "data_noise_amp": 0,
        "data_dropout": "None",

        # Model:
        "latent_multi_step": None,
        "padding_mode": "zeros",
        "latent_noise_amp": 0,
        "decoder_last_act_name": "linear",
        "hinge": 1,
        "contrastive_rel_coef": 0,
        "n_conv_layers_latent": 1,
        "is_latent_flatten": True,
        "channel_mode": "exp-16",
        "no_latent_evo": False,
        "reg_type": "None",
        "reg_coef": 0,
        "is_reg_anneal": True,
        "forward_type": "Euler",
        "evo_groups": 1,
        "evo_conv_type": "cnn",
        "evo_pos_dims": -1,
        "evo_inte_dims": -1,
        "decoder_act_name": "None",
        "vae_mode": "None",
        "uncertainty_mode": "None",

        # EBM:
        "ebm_train_mode": "cd",
        "ebm_supervised_loss_type": "mse",
        "ebm_phi_L1_coef": 0.,

        # Training:
        "is_pretrain_autoencode": False,
        "is_vae": False,
        "epochs_pretrain": 0,
        "dp_mode": "None",
        "latent_loss_normalize_mode": "None",
        "reinit_mode": "None",
        "is_clip_grad": False,
        "multi_step_start_epoch": 0,
        "epsilon_latent_loss": 0,
        "test_interval": 1,
        "lr_min_cos": 0,
        "is_prioritized_dropout": False,

        # For discriminator:
        "disc_coef": 0,
        "disc_mode": "concat",
        "disc_lr": -1,
        "disc_reg_type": "snn",
        "disc_iters": 5,
        "disc_loss_type": "hinge",
        "disc_t": "None",

        # Unet:
        "unet_fmaps": 64,
        
        #Plasma Hybrid:
        "zero_weight": 1,
        "is_mnt_indi_normalize": False,
        "noise_amp": 0,
        "sample_gaussian_mode": "diag",


        #RL:
        "processor_aggr":"max",
        "fix_alt_evolution_model":False,
        "test_reward_random_sample":False,
        
    }
    for key, item in default_param.items():
        if key not in Dict:
            Dict[key] = item
    if "seed" in Dict and Dict["seed"] is None:
        Dict["seed"] = -1
    return Dict