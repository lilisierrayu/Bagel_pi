#!/bin/bash
#SBATCH --cpus-per-task=11
#SBATCH --error=/mnt/weka/slurm_logs/lucy/img_edit_train/%j_%a_log.err
#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/weka/slurm_logs/lucy/img_edit_train/%j_%a_log.out
#SBATCH --signal=USR2@90
#SBATCH --wckey=submitit
#SBATCH --job-name=bagel
#SBATCH --qos=hl

# Check if config name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_name>"
    echo "Example: $0 seedp1_0.2_arx_biarm_allview_endspan"
    exit 1
fi

# Get config name from command line argument
config_name=$1

# Rename the job to use the config name
scontrol update job $SLURM_JOB_ID name=bagel_$config_name

cd /home/lucy/Bagel_pi
source .venv/bin/activate

# replace the variables with your own
# Fine-tuning
num_nodes=$SLURM_NNODES
node_rank=$SLURM_NODEID
master_addr=localhost
master_port=29515
model_path=/home/liliyu/workspace/BAGEL/pretrained_models/BAGEL-7B-MoT
resume_from=/mnt/weka/checkpoints/liliyu/bagel_ckpt/pre09_seed_blip3o_robot_448px_t1.0_gpu16_seq32768_shard8__PRE02_16k/checkpoints/0030000
ckpt_dir=/mnt/weka/checkpoints/lucy/bagel_ckpt/
GPUS=8
note="from_pretrained_hires_30k_2"

batch_size=1
seq_len=16384
export PYTHONPATH=/home/lucy/Bagel_pi
total_gpus=$((num_nodes * GPUS))

# Fine-tuning
srun torchrun --nnodes=$num_nodes --nproc_per_node=$GPUS \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:$master_port  train/pretrain_unified_navit.py \
  --layer_module Qwen2MoTDecoderLayer \
  --model_path $model_path \
  --resume_from $resume_from \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --exp_checkpoint_dir $ckpt_dir \
  --checkpoint_dir $ckpt_dir \
  --finetune-from-ema False \
  --log_every 1 \
  --lr 1e-5 \
  --num_worker 1 \
  --expected_num_tokens $seq_len \
  --max_num_tokens $seq_len \
  --max_num_tokens_per_sample $seq_len \
  --batch_size $batch_size \
  --dataset_config_file data/configs/${config_name}.yaml  \
  --exp_name ${config_name}_gpu${total_gpus}_${note} \
  --wandb_runid 0 \
  --num_shard $total_gpus \
  --use_flex True \
  --visual_und False \
  --save_every 1000
