MODEL_NAME="bharatgpt-3b"
DATASET_NAME="ben_tqa"
DATA_PATH="TSV_"$MODEL_NAME"_"$DATASET_NAME"_9"
BATCH_SIZE=16
SOURCE_LANGUAGE="ben_tqa"
EXTERNAL_CENTROIDS_PATH="TSV_"$MODEL_NAME"_"$SOURCE_LANGUAGE"_9/centroids.pt"
EXTERNAL_CHECKPOINT_PATH="TSV_"$MODEL_NAME"_"$SOURCE_LANGUAGE"_9/tsv_checkpoint.pt"

set -a && source .env && pixi run python tsv_main.py \
    --test \
    --model-name $MODEL_NAME \
    --dataset-name $DATASET_NAME \
    --most-likely \
    --batch-size $BATCH_SIZE \
    --source-language $SOURCE_LANGUAGE \
    --external-centroids-path $EXTERNAL_CENTROIDS_PATH \
    --external-checkpoint-path $EXTERNAL_CHECKPOINT_PATH \
    > $DATA_PATH/test.log 2>&1 &
TEST_PID=$!
echo "Process started with PID: $TEST_PID"

