# run train_t5 script
env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train_t5.py --project_name experiment1 --epochs 10 --batch_size 64;
# train_t5_learnable_query # --num_query 1
# train_t5
# train_vicuna # doesn't work. CUDA out of memory
# train_vit

# --nproc_per_node: # GPUs

# background run: nohup python -m torch.distributed.run --nproc_per_node=4 train_t5_learnable_query.py --project_name experiment1 --epochs 10 --batch_size 64 > t5_learnable_query.txt 2>&1 &