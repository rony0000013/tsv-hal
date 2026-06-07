MODEL_NAME="tiny-aya-global-3b"
DATASET_NAME="ben_tqa"
DATA_PATH="TSV_"$MODEL_NAME"_"$DATASET_NAME"_9"
BATCH_SIZE=32
set -a && source .env && pixi run python tsv_main.py \
    --train \
    --model-name $MODEL_NAME \
    --dataset-name $DATASET_NAME \
    --most-likely \
    --thres-percentile 50 \
    --batch-size $BATCH_SIZE \
    > $DATA_PATH/train.log 2>&1 &
TRAIN_PID=$!
echo "Process started with PID: $TRAIN_PID"
