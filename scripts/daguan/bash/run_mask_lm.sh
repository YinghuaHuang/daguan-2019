python -m opennmt.bin.main train_and_eval \
       --model_type TransformerMaskLM \
       --config scripts/daguan/config/daguan_mask_lm.yml \
       --gpu_allow_growth \
       --num_gpus 3
