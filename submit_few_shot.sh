#!/bin/bash
#SBATCH --job-name=few_shot_base
#SBATCH --output=logs/out/%x-%j.out
#SBATCH --error=logs/err/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-user=mustun@bio.ens.psl.eu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=ioc@v100  


module purge
module load pytorch-gpu/py3/2.0.0
conda activate beans

cd $WORK/beans 

python scripts/few_shot_dolphin_reef.py \
  --samples-per-class 50 100 200 \
  --model-type dolph2vec \
  --dolph2vec-variant base \
  --classifier-type linear \
  --task classification \
  --output-dir logs/few_shot/dolph2vec_base \
  --batch-size 32 \
  --lrs "[0.0001, 0.00005, 0.00001]" \
  --epochs 30 

