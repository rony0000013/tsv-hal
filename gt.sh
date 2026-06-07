MODEL_NAME="tiny-aya-global-3b"
DATASET_NAME="ben_tqa"
DATA_PATH="TSV_"$MODEL_NAME"_"$DATASET_NAME"_9"
set -a && source .env && pixi run python tsv_main.py \
    --generate-gt \
    --model-name $MODEL_NAME \
    --dataset-name $DATASET_NAME \
    --most-likely \
    --num-gene 1 \
    > $DATA_PATH/gt.log 2>&1 &
GT_PID=$!
echo "Process started with PID: $GT_PID"
