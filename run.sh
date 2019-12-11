#!/bin/bash -l
# SBATCH --nodes=1
# SBATCH --time=72:0:0
# SBATCH --qos=gpu
# SBATCH --gres=gpu:8
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=1
# SBATCH --partition=gpu
# SBATCH -m=40G

module purge
slmodules -s x86_E5v2_Mellanox_GPU
module load gcc cuda cudnn mvapich2 openblas
source ~/venvs/tensorflow-1.9/bin/activate

srun python train_classification_tl.py --ckpt_load=keras_model_untrained.h5 --epochs=1000 --batch_size=100 --epochs_ckpt=20 --train_set=train_0_7.pickle --test_set=test_0_7.pickle --validation_split=0.25 --from_scratch=False --skip_train=False --skip_test=False --optimizer=rmsprop --loss=binary_crossentropy --verbose=1

deactivate