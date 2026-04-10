MODEL_NAME="sarvam-1"
DATASET_NAME="hindi_tqa"
DATA_PATH="TSV_"$MODEL_NAME"_"$DATASET_NAME"_9"
BATCH_SIZE=8
set -a && source .env && pixi run python tsv_main.py --test --model-name $MODEL_NAME --dataset-name $DATASET_NAME --most-likely --batch-size $BATCH_SIZE > $DATA_PATH/test.log 2>&1 &
TEST_PID=$!
echo "Process started with PID: $TEST_PID"

