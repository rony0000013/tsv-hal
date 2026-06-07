MODEL_NAME="llama-3.2-3b"
SOURCE_LANGUAGE="tqa" # Source
DATASET_NAME="tqa" # Destination
DATA_PATH="TSV_"$MODEL_NAME"_"$DATASET_NAME"_9"
BATCH_SIZE=32
EXTERNAL_CENTROIDS_PATH="TSV_"$MODEL_NAME"_"$SOURCE_LANGUAGE"_9/centroids.pt"
EXTERNAL_CHECKPOINT_PATH="TSV_"$MODEL_NAME"_"$SOURCE_LANGUAGE"_9/tsv_checkpoint.pt"

echo "Source :" $SOURCE_LANGUAGE "| Destination :" $DATASET_NAME
set -a && source .env && pixi run python tsv_main.py \
    --test \
    --model-name $MODEL_NAME \
    --dataset-name $DATASET_NAME \
    --most-likely \
    --batch-size $BATCH_SIZE \
    --thres-percentile 50 \
    --source-language $SOURCE_LANGUAGE \
    --external-centroids-path $EXTERNAL_CENTROIDS_PATH \
    --external-checkpoint-path $EXTERNAL_CHECKPOINT_PATH \
    > $DATA_PATH/test.log 2>&1

cat $DATA_PATH/test.log | rg "Test AUROC:"
