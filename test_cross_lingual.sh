#!/bin/bash

# # Check if model name is provided
# if [ -z "$1" ]; then
#     echo "Usage: ./run_evals.sh <model_name>"
#     echo "Example: ./run_evals.sh llama-3.2-3b"
#     exit 1
# fi

MODEL_NAME="nanda-10b"
BATCH_SIZE=32
DATASETS=("tqa" "hindi_tqa" "ben_tqa")

# Array to store results for the final summary
declare -A RESULTS

echo "=================================================="
echo "Starting evaluation for Model: $MODEL_NAME"
echo "=================================================="

# Loop through Source datasets
for SOURCE_LANGUAGE in "${DATASETS[@]}"; do
    # Loop through Destination datasets
    for DATASET_NAME in "${DATASETS[@]}"; do
        
        # Define paths dynamically
        DATA_PATH="TSV_${MODEL_NAME}_${DATASET_NAME}_9"
        EXTERNAL_CENTROIDS_PATH="TSV_${MODEL_NAME}_${SOURCE_LANGUAGE}_9/centroids.pt"
        EXTERNAL_CHECKPOINT_PATH="TSV_${MODEL_NAME}_${SOURCE_LANGUAGE}_9/tsv_checkpoint.pt"
        
        # Ensure the output log directory exists
        mkdir -p "$DATA_PATH"
        
        echo "Running Source: $SOURCE_LANGUAGE -> Destination: $DATASET_NAME ..."
        
        # Run the evaluation command
        set -a && source .env && pixi run python tsv_main.py \
            --test \
            --model-name "$MODEL_NAME" \
            --dataset-name "$DATASET_NAME" \
            --most-likely \
            --batch-size $BATCH_SIZE \
            --thres-percentile 50 \
            --source-language "$SOURCE_LANGUAGE" \
            --external-centroids-path "$EXTERNAL_CENTROIDS_PATH" \
            --external-checkpoint-path "$EXTERNAL_CHECKPOINT_PATH" \
            > "$DATA_PATH/test.log" 2>&1
            
        # Extract the AUROC value using ripgrep (rg) or grep as fallback
        if command -v rg &> /dev/null; then
            AUROC_VAL=$(cat "$DATA_PATH/test.log" | rg "Test AUROC:" | awk -F' ' '{print $3}')
        else
            AUROC_VAL=$(cat "$DATA_PATH/test.log" | grep "Test AUROC:" | awk -F' ' '{print $3}')
        fi
        
        # If AUROC wasn't found (e.g. script crashed), default to "N/A"
        if [ -z "$AUROC_VAL" ]; then
            AUROC_VAL="ERROR/NA"
        fi
        
        # Store the result in our associative array
        RESULTS["${SOURCE_LANGUAGE}_to_${DATASET_NAME}"]=$AUROC_VAL
        
    done
done

# -----------------------------------------------------------------
# Print Final Summary
# -----------------------------------------------------------------
echo -e "\n=================================================="
echo "                 FINAL RESULTS                    "
echo "=================================================="

echo -e "\n--- The 6 Cross-Dataset Transfer Values ---"
for SOURCE_LANGUAGE in "${DATASETS[@]}"; do
    for DATASET_NAME in "${DATASETS[@]}"; do
        if [ "$SOURCE_LANGUAGE" != "$DATASET_NAME" ]; then
            echo "Source: $SOURCE_LANGUAGE ➔ Destination: $DATASET_NAME | AUROC: ${RESULTS["${SOURCE_LANGUAGE}_to_${DATASET_NAME}"]}"
        fi
    done
done

echo -e "\n--- Same-Dataset Base Values (For reference) ---"
for DATASET in "${DATASETS[@]}"; do
    echo "Source: $DATASET ➔ Destination: $DATASET | AUROC: ${RESULTS["${DATASET}_to_${DATASET}"]}"
done
echo "=================================================="