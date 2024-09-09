#!/bin/bash
#SBATCH --account=rrg-menna
#SBATCH --time=03:00:00                   # The job will run for 3 hours
#SBATCH --cpus-per-task=4                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --output="video.%j.%N.out"
#SBATCH --error=errors/video-%A%a.err

 # 1. Create your environment locally

CURDIR=/home/liila/scratch/medvt/

# Load required modules
#module load python/3.10
#module load cuda/12.2

# 1. Create a virtual environment
#python3 -m venv ${CURDIR}/myenv

# 2. Activate the virtual environment
source ${CURDIR}/myenv/bin/activate

# 3. Install dependencies from requirements.txt
#pip install --no-index  -r ${CURDIR}/requirements.txt

#pip install --no-index numpy  torch torchvision

#pip install --no-index scipy tqdm pandas cython  scikit-learn opencv-python matplotlib pillow imageio scikit-image einops ipdb resampy soundfile torch torchvision
#pip install opencv-python

# 4. Unzip dataset
#unzip dataset.zip

# 5. Run train_on_davis and kittimots
#python3 ${CURDIR}/avos/train_medvtmm_avos.py --output_dir ${CURDIR}/outputs/medvtmm/medvt_avos/train --pvt_weights_path ${CURDIR}/pretrained_backbones/avsbench/pvt_v2_b5.pth --swin_b_pretrained_path ${CURDIR}/pretrained_backbones/avos/swin_base_patch244_window877_kinetics400_22k.pth --resnet101_coco_weights_path ${CURDIR}/pretrained_backbones/avos/384_coco_r101.pth --use_flow=0 

# 6. Run inference_on_kittimots
#ckpts kitti
#python3 ${CURDIR}/avos/inference_swin_medvt_avos.py --model_path ${CURDIR}/outputs/medvtmm/medvt_avos/train/medvtmm_20240531083217_medvt_swinB_df2048_enc61_dec12layers4scales_t360v360f6_flow0_mmfuseattnqgen1_lppmode2_lppsc8.0_lr1.0e-04_1.0e-05_aux0.5_ep15/checkpoint_best.pth --dataset kittimots --val_size 473 --flip --msc --outputs ${CURDIR}/outputs/swin_medvt/kittimots --save_pred --save_gt_overlay

#ckpts davis
#python3 ${CURDIR}/avos/inference_swin_medvt_avos.py --model_path ${CURDIR}/outputs/medvtmm/medvt_avos/train/medvtmm_20240607161628_medvt_swinB_df2048_enc61_dec12layers4scales_t360v360f6_flow0_mmfuseattnqgen1_lppmode2_lppsc8.0_lr1.0e-04_1.0e-05_aux0.5_ep15/checkpoint_best.pth --dataset kittimots --val_size 473 --flip --msc --outputs ${CURDIR}/outputs/swin_medvt/kittimots --save_pred --save_gt_overlay

# 7. Run inference_on_davis
#python3 ${CURDIR}/avos/inference_swin_medvt_avos.py --model_path ${CURDIR}/outputs/medvtmm/medvt_avos/train/medvtmm_20240607161628_medvt_swinB_df2048_enc61_dec12layers4scales_t360v360f6_flow0_mmfuseattnqgen1_lppmode2_lppsc8.0_lr1.0e-04_1.0e-05_aux0.5_ep15/checkpoint_best.pth --dataset davis --val_size 360 --flip --msc --outputs ${CURDIR}/outputs/swin_medvt/davis --save_pred --save_gt_overlay

#8. Run inference_on_bdd
#ckpt kitti
#python3 ${CURDIR}/avos/inference_swin_medvt_avos.py --model_path ${CURDIR}/outputs/medvtmm/medvt_avos/train/medvtmm_20240531083217_medvt_swinB_df2048_enc61_dec12layers4scales_t360v360f6_flow0_mmfuseattnqgen1_lppmode2_lppsc8.0_lr1.0e-04_1.0e-05_aux0.5_ep15/checkpoint_best.pth --dataset bdd --val_size 700 --flip --msc --outputs ${CURDIR}/outputs/swin_medvt/bdd --save_pred --save_gt_overlay
#ckpts bdd
python3 ${CURDIR}/avos/inference_swin_medvt_avos.py --model_path ${CURDIR}/outputs/medvtmm/medvt_avos/train/medvtmm_20240613095131_medvt_swinB_df2048_enc61_dec12layers4scales_t360v360f6_flow0_mmfuseattnqgen1_lppmode2_lppsc8.0_lr1.0e-04_1.0e-05_aux0.5_ep15/checkpoint_best.pth --dataset bdd --val_size 700 --flip --msc --outputs ${CURDIR}/outputs/swin_medvt/bdd --save_pred --save_gt_overlay
#ckpts davis
#python3 ${CURDIR}/avos/inference_swin_medvt_avos.py --model_path ${CURDIR}/outputs/medvtmm/medvt_avos/train/medvtmm_20240607161628_medvt_swinB_df2048_enc61_dec12layers4scales_t360v360f6_flow0_mmfuseattnqgen1_lppmode2_lppsc8.0_lr1.0e-04_1.0e-05_aux0.5_ep15/checkpoint_best.pth --dataset bdd --val_size 700 --flip --msc --outputs ${CURDIR}/outputs/swin_medvt/bdd --save_pred --save_gt_overlay
