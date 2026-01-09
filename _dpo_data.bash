#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=cmv-dataset-dpo-data
#SBATCH --output=log/%j-%x.out

CMV_DATASET="data/cmv_data.jsonl"

python 20_dpo_data.py $CMV_DATASET data/dpo_data.jsonl
