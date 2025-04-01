PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m train \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_lora_modules 100 \
    --lora_modules_truncate_tail True \
    --lora_modules_truncate_offset 100 \
    --freeze_vision_tower True \
    --dataset dataset/train.jsonl \
    --output_dir output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --projector_lr 1e-5 \
    --train_projector False \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --dataloader_num_workers 4 \
    --seed 42
