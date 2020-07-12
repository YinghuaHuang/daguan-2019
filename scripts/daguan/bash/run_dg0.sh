python -m opennmt.bin.main train_and_eval \
       --model_type TransformerSeqTagger \
       --checkpoint_path ckpt/daguan/mask_lm \
       --source_scope transformer_mask_lm/encoder \
       --target_scope transformer_sequence_tagger/encoder \
       --config scripts/daguan/config/daguan_k_fold_0.yml \
       --gpu_allow_growth \
       --num_gpus 1
