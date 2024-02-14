# LE-PDE-UQ: Uncertainty Quantification for Forward and Inverse Problems of PDEs via Latent Global Evolution

[Paper](https://arxiv.org/abs/2206.07681) | [Poster](https://github.com/snap-stanford/le_pde/blob/master/assets/lepde_poster.pdf) | [Slide](https://docs.google.com/presentation/d/1Qgbd_vVbFAnjqkvIH8p_t9mfUQRWKr1ZAkxzzavoGhc/edit?usp=share_link)

Official repo for the paper [Uncertainty Quantification for Forward and Inverse Problems of PDEs via Latent Global Evolution](https://arxiv.org/abs/2206.07681) </br>
[Tailin Wu](https://tailin.org/), [Willie Neiswanger](https://willieneis.github.io/), Hongtao Zheng, [Stefano Ermon](https://cs.stanford.edu/~ermon/), [Jure Leskovec](https://cs.stanford.edu/people/jure/) </br>
AAAI 2024 **Oral**</br>


Our method, termed Latent Evolution of PDEs with Uncertainty Quantification (LE-PDE-UQ), endows deep learning-based surrogate models with robust and efficient uncertainty quantification capabilities for both forward and inverse problems.

In extensive experiments, we demonstrate the accurate uncertainty quantification performance of our approach, surpassing that of strong baselines including deep ensembles, Bayesian neural network layers, and dropout. Our method excels at propagating uncertainty over extended auto-regressive rollouts, making it suitable for scenarios involving long-term predictions.

<a href="url"><img src="./assert/le_pde_uq.png" align="center" width="700" ></a>

# Installation

1. First clone the directory. Then run the following command to initialize the submodules:

```code
git submodule init; git submodule update
```
(If showing error of no permission, need to first [add a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).)

2. Install dependencies.

First, create a new environment using [conda](https://docs.conda.io/en/latest/miniconda.html) (with python >= 3.7). Then install pytorch, torch-geometric and other dependencies as follows (the repository is run with the following dependencies. Other version of torch-geometric or deepsnap may work but there is no guarentee.)

Install pytorch (replace "cu113" with appropriate cuda version. For example, cuda11.1 will use "cu111"):
```code
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Install torch-geometric. Run the following command:
```code
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-geometric==1.7.2
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
```

Install other dependencies:
```code
pip install -r requirements.txt
```

If you want to do inverse optimization, run also the following command inside the root directory of PDE_Control repository:
```code
pip install PDE-Control/PhiFlow/[gui] jupyterlab
```


# Dataset

The dataset files can be downloaded via [this link](https://drive.google.com/drive/folders/1rwcnT0g4_MiZfYUU4y7ybnfk8d4qgMEg?usp=share_link). 
* Download the files under "fno_data/" in the link into the "data/fno_data/" folder in the local repo.
# Training

Below we provide example commands for training LE-PDE-UQ. For all the commands that reproduce the experiments in the paper.

## Forward Problems:
### Table1 in Our Paper:
Bayes layer (with latent)
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=bayeslayer --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=new
```

Bayes layer (without latent)
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=True --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=bayeslayer --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=new --seed=1
```

Dropout, L2=0
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=dropout:0.5 --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=new
```

Dropout, L2=1e-5
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=dropout:0.5 --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=1e-5 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=new
```

Dropout, L2=1e-4
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=dropout:0.5 --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=1e-4 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=new
```

Dropout, L2=1e-3
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=dropout:0.5 --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=1e-3 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=new
```

NoLatent
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=tailin-uncertainty --date_time=2022-10-09 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --decoder_act_name=silu --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=512 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mpe-1.5 --uncertainty_mode=diag^sep:256 --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=5 --id=best_rerun
```

Latent (single, without $\sigma$)
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=256 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=0
```

Latent 
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=diag^sub:256 --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=new --seed=10 
```

### Table2 in Our Paper:
NoLatent (ensemble, with $\sigma$)
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=True --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=diag^full --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=0 --seed=10 
```
Latent (ensemble, with $\sigma$)
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=diag^full --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=new --seed=10 
```

### Table3 in Our Paper:
NoLatent (ensemble, with $\sigma$)
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=True --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=diag^samplefull --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=0 --seed=10 
```

Latent (ensemble, with $\sigma$)
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=diag^samplefull --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --seed=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=0 --id=new --seed=10 
```

### Key Factors Influence:
Latent full + L1
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=l1 --uncertainty_mode=diag^sub:256 --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=1 --id=fifth_new --seed=10 
```

## Inverse Problem
Latent(ours, ensemble, with $\sigma$)
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=False --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=diag^sub:256 --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=7 --id=3rd_new --seed=10 
```

NoLatent(ours, ensemble, with $\sigma$)
```code
export OMP_NUM_THREADS=6; python3 train.py --exp_id=anonymous-uncertainty --date_time=2023-08-19 --dataset=fno-4 --n_train=-1 --time_interval=1 --save_interval=10 --algo=contrast --reg_type=None --reg_coef=0 --is_reg_anneal=True --no_latent_evo=True --encoder_type=cnn-s --input_steps=10 --evolution_type=mlp-3-elu-2 --decoder_type=cnn-tr --encoder_n_linear_layers=0 --n_conv_blocks=4 --n_latent_levs=1 --n_conv_layers_latent=3 --channel_mode=exp-16 --is_latent_flatten=True --evo_groups=1 --recons_coef=1 --consistency_coef=1 --contrastive_rel_coef=0 --hinge=0 --density_coef=0.001 --latent_noise_amp=1e-5 --normalization_type=gn --latent_size=384 --kernel_size=4 --stride=2 --padding=1 --padding_mode=zeros --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --latent_multi_step=1^2^3^4 --use_grads=False --use_pos=False --is_y_diff=False --loss_type=mse --uncertainty_mode=diag^sub:256 --loss_type_consistency=mse --batch_size=20 --val_batch_size=20 --epochs=200 --opt=adam --weight_decay=0 --disc_coef=0 --id=0 --verbose=1 --save_iterations=400 --latent_loss_normalize_mode=targetindi --n_workers=0 --gpuid=7 --id=3rd_new --seed=10 
```

# Analysis
All the analysis codes we used can be found in the le_pde_uq_analysis.ipynb notebook.

# Related Projects:

* [LAMP](https://github.com/snap-stanford/lamp) (ICLR 2023 spotlight): first fully DL-based surrogate model that jointly optimizes spatial resolutions to reduce computational cost and learns the evolution model, learned via reinforcement learning.

* [LE-PDE](https://github.com/snap-stanford/le_pde) (NeurIPS 2022): It introduces a simple, fast and scalable LE-PDE method to accelerate the simulation and inverse optimization of PDEs, which are crucial in many scientific and engineering applications (e.g., weather forecasting, material science, engine design).
# Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{wu2024uncertainty,
title={Uncertainty Quantification for Forward and Inverse Problems of PDEs via Latent Global Evolution},
author={Wu, Tailin and Neiswanger, Willie and Zheng, Hongtao and Ermon, Stefano and Leskovec, Jure},
booktitle={Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence},
year={2024},
}
```
