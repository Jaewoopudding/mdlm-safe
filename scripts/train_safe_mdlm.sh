CUDA_VISIBLE_DEVICES=0,2 python -u -m main \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  model=small \
  data=safe \
  wandb.name=mdlm-safe \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  loader.global_batch_size=1024 \
  optim.lr=3e-4 \
  trainer.max_steps=50000 \
  noise=loglinear \
  optim.beta1=0.9 \
  optim.beta2=0.999 \
  trainer.devices=2 \
  backbone=bert \
  checkpointing.resume_ckpt_path=/home/jaewoo/research/mdlm/outputs/datamol-io/safe-gpt/2025.04.06/170126/checkpoints/last.ckpt \



# max_position_embeddings=256
# vocab_size=1880
# batch_size=2048
# learning_rate = 3e-4
# num_train_steps = 50000
# schedule loglinear
# adamw (0.9, 0.999)