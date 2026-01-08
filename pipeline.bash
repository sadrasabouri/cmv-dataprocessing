#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
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

SUBMISSIONS_W_DELTA="~ids.txt"
echo "01. Extracting uniq id for delta posts ... [saving into $SUBMISSIONS_W_DELTA]"
tail -n +2 $DELTA | cut -d, -f1 | sort | uniq > $SUBMISSIONS_W_DELTA

CMV_DELTA_FILTERED_SUB="data/changemyview_submissions.jsonl"
echo "02. Filtering the posts with those ids ... [saving into $CMV_DELTA_FILTERED_SUB]"
grep -Ff <(sed 's/.*/"id":"&"/' $SUBMISSIONS_W_DELTA) $CMV_SUB > $CMV_DELTA_FILTERED_SUB 

CMV_DELTA_FILTERED_COMMENT="data/changemyview_comments.jsonl"
echo "03. Filtering the comments with those ids ... [saving into $CMV_DELTA_FILTERED_COMMENT]"
grep -Ff <(sed 's/^/link_id":"t3_/; s/$/"/' $SUBMISSIONS_W_DELTA) $CMV_COMMENTS > $CMV_DELTA_FILTERED_COMMENT

echo "10. Making the dataset ..."
CMV_DATASET="data/cmv_data.jsonl"
python 10_make_dataset.py $CMV_DELTA_FILTERED_SUB $CMV_DELTA_FILTERED_COMMENT $DELTA $CMV_DATASET

echo "20. OP responses w/wo delta"
python 20_op_responses.py $CMV_DELTA_FILTERED_SUB $CMV_DELTA_FILTERED_COMMENT $DELTA --use-cache
