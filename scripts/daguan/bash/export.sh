python -m opennmt.bin.main export \
          --model_type TransformerSeqTagger \
          --config scripts/daguan/config/daguan/daguan_k_fold_4.yml \
          --checkpoint_path /data/public/yinghua/code/OpenNMT-tf/ckpt/daguan_k_fold_trans/4 \
          --gpu_allow_growth --num_gpus 1
