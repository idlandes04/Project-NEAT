@echo off
REM Windows batch script to train NEAT model on 3080ti

echo Training NEAT model on Windows with 3080ti GPU...

python main.py --mode train ^
    --use_titans_memory ^
    --use_transformer2_adaptation ^
    --use_mvot_processor ^
    --use_blt_processor ^
    --blt_checkpoint_path ./outputs/blt/mock_byte_lm.pt ^
    --mvot_codebook_path ./outputs/mvot/mock_codebook.pt ^
    --hidden_size 768 ^
    --num_layers 12 ^
    --num_attention_heads 12 ^
    --batch_size 16 ^
    --learning_rate 5e-5 ^
    --max_steps 10000 ^
    --gradient_accumulation_steps 1 ^
    --mixed_precision ^
    --gradient_checkpointing ^
    --entropy_threshold 0.5 ^
    --output_dir ./outputs/neat_model_full

echo Training complete!
