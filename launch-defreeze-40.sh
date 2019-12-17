#!/bin/bash -l

##SBATCH --account=leso-pb

## specific for GPU nodes (free account)
#SBATCH --partition=gpu --qos=gpu --gres=gpu:4

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=72:00:00

echo STARTING AT $(date)

module purge
source /ssoft/spack/bin/slmodules.sh -s x86_E5v2_Mellanox_GPU
module load gcc cuda cudnn python mvapich2
source /home/amjakob/venvs/tensorflow-1.9/bin/activate

time srun python train_classification_tl_ft.py --ckpt_load=keras_swisspv_untrained.h5 --epochs=500 --batch_size=128 --epochs_ckpt=5 --train_set=train.pickle --test_set=test.pickle --validation_split=0.25 --from_scratch=True --skip_train=False --skip_test=False --optimizer=rmsprop --loss=binary_crossentropy --fine_tuning=True --fine_tune_layers=40 --verbose=True

deactivate

echo FINISHED AT $(date)