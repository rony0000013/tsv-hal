MODEL_NAME="nanda-10b"
DATASET_NAME="ben_tqa"
DATA_PATH="TSV_"$MODEL_NAME"_"$DATASET_NAME"_9"
mkdir -p $DATA_PATH
set -a && source .env && pixi run python tsv_main.py \
    --gene \
    --model-name $MODEL_NAME \
    --dataset-name $DATASET_NAME \
    --most-likely \
    --num-gene 1 \
    > $DATA_PATH/gen.log 2>&1 &
GEN_PID=$!
echo "Process started with PID: $GEN_PID"
