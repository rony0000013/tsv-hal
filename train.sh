CUDA_VISIBLE_DEVICES=0 pixi run python tsv_main.py --train --model-name sarvam-1  --dataset-name hindi_tqa --most-likely --batch-size 8 > train.log 2>&1 &

