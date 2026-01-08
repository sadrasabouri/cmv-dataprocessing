#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=cmv-dataset-pipeline
#SBATCH --output=log/%j-%x.out

DATA_DIR="../cmv/reddit/subreddits24"
DELTA_LOG_SUB="$DATA_DIR/DeltaLog_submissions"
CMV_SUB="$DATA_DIR/changemyview_submissions"
CMV_COMMENTS="$DATA_DIR/changemyview_comments"

echo "00. Extracting deltas ..."
DELTA="data/deltas.csv"
python 00_extract_deltas.py $DELTA_LOG_SUB $DELTA

echo "10. Making the dataset ..."
CMV_DATASET="data/cmv_data.jsonl"
python 10_make_dataset.py $CMV_SUB $CMV_COMMENTS $DELTA $CMV_DATASET

echo "20. OP responses w/wo delta"
python 20_op_responses.py $CMV_SUB $CMV_COMMENTS $DELTA --use-cache
