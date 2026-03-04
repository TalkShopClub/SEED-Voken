python evaluation_image.py \
  --config_file configs/IBQ/gpu/finetune_256_ocr_smart_resize.yaml \
  --ckpt_path ../../checkpoints/vqgan/ibq_finetune_ocr_smart_resize/last.ckpt \
  --model IBQ \
  --batch_size 1 \
  --save_comparison_dir /workspace/visualize \
  --image_size 0.5 \
  --skip_oom \