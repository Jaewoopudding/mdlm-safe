CUDA_VISIBLE_DEVICES=4 python -u -m main \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small \
  data=safe \
  wandb.name=mdlm-safe \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  loader.global_batch_size=2048 \
  optim.lr=3e-4 \
  trainer.max_steps=50000 \
  noise=loglinear \
  optim.beta1=0.9 \
  optim.beta2=0.999 \
  trainer.devices=1 \
  backbone=bert \


# max_position_embeddings=256
# vocab_size=1880
# batch_size=2048
# learning_rate = 3e-4
# num_train_steps = 50000
# schedule loglinear
# adamw (0.9, 0.999)