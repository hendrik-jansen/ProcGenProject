#!/bin/bash
#SBATCH --job-name=mg_25b_paired
# give it any name you want
#SBATCH --cpus-per-task=4
# max 24 per node

#SBATCH --partition=week
# choose out of day, week, month depending on job duration
#SBATCH --mem-per-cpu=32G
# max 251GB per node
#SBATCH --gres=gpu:1
# how many gpus to use
# each node has 4 gpus
#SBATCH --time=72:00:00
# job length: the job will run either until completion or until this timer runs out
#SBATCH --error=job.%J.err
# %J is the job ID, errors will be written to this file
#SBATCH --output=job.%J.out
# the output will be written in this file
#SBATCH --mail-type=FAIL
# write a mail if a job begins, ends, fails, gets requeued or stages out
# options: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=h.jansen@student.uni-tuebingen.de
# your email
# here will be your commands for running the script
singularity run ~/container/container_dcd.sif python3.8 dcd/train.py \
    --xpid=ued-MultiGrid-GoalLastFewerBlocksAdversarial-v0-paired-lstm256ae-lr0.0001-epoch5-mb1-v0.5-gc0.5-henv0.0-ha0.0-tl_0 \
    --env_name=MultiGrid-GoalLastFewerBlocksAdversarial-v0 \
    --use_gae=True \
    --gamma=0.995 \
    --gae_lambda=0.95 \
    --seed=88 \
    --num_control_points=12 \
    --recurrent_arch=lstm \
    --recurrent_agent=True \
    --recurrent_adversary_env=True \
    --recurrent_hidden_size=256 \
    --use_global_critic=False \
    --lr=0.0001 \
    --num_steps=256 \
    --num_processes=32 \
    --num_env_steps=250000000 \
    --ppo_epoch=5 \
    --num_mini_batch=1 \
    --entropy_coef=0.0 \
    --value_loss_coef=0.5 \
    --clip_param=0.2 \
    --clip_value_loss=True \
    --adv_entropy_coef=0.0 \
    --max_grad_norm=0.5 \
    --algo=ppo \
    --ued_algo=paired \
    --use_plr=False \
    --level_replay_prob=0.0 \
    --level_replay_rho=1.0 \
    --level_replay_seed_buffer_size=5000 \
    --level_replay_score_transform=rank \
    --level_replay_temperature=0.1 \
    --staleness_coef=0.3 \
    --no_exploratory_grad_updates=False \
    --use_editor=False \
    --level_editor_prob=0 \
    --level_editor_method=random \
    --num_edits=0 \
    --base_levels=batch \
    --use_accel_paired=False \
    --accel_paired_score_function=paired \
    --use_lstm=False \
    --use_behavioural_cloning=False \
    --kl_loss_coef=0.0 \
    --kl_update_step=1 \
    --use_kl_only_agent=False \
    --log_interval=25 \
    --screenshot_interval=1000 \
    --log_grad_norm=False \
    --handle_timelimits=True \
    --log_dir=~/logs/dcd_paired \
    --log_action_complexity=True \
    --archive_interval=30518 \
    --log_plr_buffer_stats=True \
    --log_replay_complexity=True \
    --reject_unsolvable_seeds=False \
    --checkpoint=True
