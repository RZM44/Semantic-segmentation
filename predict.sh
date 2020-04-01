#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Interactive
#SBATCH --gres=gpu:4
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-00:30:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:

export MPLBACKEND="agg"
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
#cd ..
#cd ..
python predict.py --batch_size 32 --num_class 21 --crop_size 64 --continue_from_epoch -2 --learn_rate 0.007 --num_epochs 52 --experiment_name trainc64b32 --use_gpu True --mementum 0.9 --weight_decay 5e-4 --output_stride 16
python predict.py --batch_size 32 --num_class 21 --crop_size 128 --continue_from_epoch -2 --learn_rate 0.007 --num_epochs 52 --experiment_name trainc128b32 --use_gpu True --mementum 0.9 --weight_decay 5e-4 --output_stride 16
python predict.py --batch_size 32 --num_class 21 --crop_size 256 --continue_from_epoch -2 --learn_rate 0.007 --num_epochs 52 --experiment_name trainc256b32 --use_gpu True --mementum 0.9 --weight_decay 5e-4 --output_stride 16
