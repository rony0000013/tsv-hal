CUDA_VISIBLE_DEVICES=0 pixi run python tsv_main.py --gene --model-name llama3.2-3B  --dataset-name hindi_tqa --most-likely --num-gene 1  > gen.log 2>&1 &

