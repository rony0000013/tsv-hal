CUDA_VISIBLE_DEVICES=0 pixi run python tsv_main.py --generate-gt --model-name sarvam-1  --dataset-name hindi_tqa --most-likely --num-gene 1  > gt.log 2>&1 &
