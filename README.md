# Eigen-MoRA: MoRA with Eigenvector based non-param projection

- Original Paepr: [MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning](https://arxiv.org/abs/2405.12130)

## Idea

- Can Non-parametric function could retrain important informations?

## Setup

```
pip install -e ./peft-mora
pip install -U trl
```

## Sample exec

```
(Eigen-MoRA) ➜  Eigen-MoRA git:(main) ✗ python train.eigen-mora.trl.py
Dataset({
    features: ['instruction', 'output', 'url', 'prompt', 'completion'],
    num_rows: 21155
})
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████| 6/6 [00:03<00:00,  1.79it/s]
LoraConfig(peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path=None, revision=None, task_type='CAUSAL_LM', inference_mode=False, r=1024, target_modules={'q_proj', 'k_proj', 'v_proj'}, lora_alpha=8, lora_dropout=0.1, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', loftq_config={}, use_dora=False, use_mora=False, use_eigenmora=True, mora_type=1)
56.014604806900024s for init model
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(128256, 4096)
        (layers): ModuleList(
          (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaSdpaAttention(
              (q_proj): lora.Linear(
                (base_layer): Linear(in_features=4096, out_features=4096, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=1024, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=1024, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (eigenmora_eigenvector_matrices): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 4096x1024 (cuda:0)])
              )
              (k_proj): lora.Linear(
                (base_layer): Linear(in_features=4096, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=1024, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=1024, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (eigenmora_eigenvector_matrices): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 1024x1024 (cuda:0)])
              )
              (v_proj): lora.Linear(
                (base_layer): Linear(in_features=4096, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=1024, out_features=1024, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=1024, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (eigenmora_eigenvector_matrices): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 1024x1024 (cuda:0)])
              )
              (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
              (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
              (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm()
            (post_attention_layernorm): LlamaRMSNorm()
          )
        )
        (norm): LlamaRMSNorm()
      )
      (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
    )
  )
)
trainable params: 100,663,296 || all params: 8,332,251,136 || trainable%: 1.2081164424470847
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
{'loss': 1.9786, 'grad_norm': 17.23695182800293, 'learning_rate': 6.024096385542169e-07, 'epoch': 0.0}                  
{'loss': 1.8865, 'grad_norm': 5.561951160430908, 'learning_rate': 6.024096385542169e-06, 'epoch': 0.0}                  
{'loss': 1.7749, 'grad_norm': 4.064998149871826, 'learning_rate': 1.2048192771084338e-05, 'epoch': 0.01}                
{'loss': 1.6799, 'grad_norm': 3.5034334659576416, 'learning_rate': 1.8072289156626505e-05, 'epoch': 0.01}               
{'loss': 1.6943, 'grad_norm': 5.886814117431641, 'learning_rate': 2.4096385542168677e-05, 'epoch': 0.01}                
{'loss': 1.7631, 'grad_norm': 3.7718868255615234, 'learning_rate': 3.012048192771085e-05, 'epoch': 0.02}                
{'loss': 2.275, 'grad_norm': 3.978529453277588, 'learning_rate': 3.614457831325301e-05, 'epoch': 0.02}                  
{'loss': 1.7087, 'grad_norm': 6.5003252029418945, 'learning_rate': 4.2168674698795186e-05, 'epoch': 0.03}               
{'loss': 1.7382, 'grad_norm': 4.098443984985352, 'learning_rate': 4.8192771084337354e-05, 'epoch': 0.03}                
{'loss': 1.9003, 'grad_norm': 9.242776870727539, 'learning_rate': 5.4216867469879516e-05, 'epoch': 0.03}                
{'loss': 1.7631, 'grad_norm': 4.730923175811768, 'learning_rate': 6.02409638554217e-05, 'epoch': 0.04}                  
{'loss': 1.8257, 'grad_norm': 3.383481025695801, 'learning_rate': 6.626506024096386e-05, 'epoch': 0.04}               
```

---

> Origianl MoRA README

## Setup

We implement MoRA in peft-mora based on HF peft in the [`apply_mora`](https://github.com/kongds/MoRA/blob/main/peft-mora/src/peft/tuners/lora/layer.py#L229) and [`get_delta_weight`](https://github.com/kongds/MoRA/blob/main/peft-mora/src/peft/tuners/lora/layer.py#L514).
``` sh
pip install -e ./peft-mora
```

After installation, it can be used like

``` python
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    # enable MoRA
    use_mora=True,
    # type 1 (Sharing) for large lora ranks, Eq. 6 in paper
    # type 6 (RoPE based) for small lora ranks, Eq. 9 in paper
    mora_type=6,
    # lora rank here, we will calculate corresponding $\hat{r}$ in MoRA
    r=lora_r,
    # MoRA does not use lora_alpha
    # lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    task_type="CAUSAL_LM",
    **kwargs,
)
model = get_peft_model(model, config)

# training here...

# can be merged into model via `merge_and_unload` like LoRA
model = model.merge_and_unload() 
```

## Examples
### fine-tuning MetaMath with MoRA

``` sh
RANK=8
deepspeed --num_gpus=8 --num_nodes=2 train.py \
           --base_model <LLAMA-2> --micro_batch_size 4\
            --wandb_run_name mora_math_r8 --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
            --num_epochs 3 --deepspeed ds.config --wandb_project lora-math --lora_r $RANK --batch_size 128 \
            --data_path meta-math/MetaMath \
            --save_steps 3000 \
            --learning_rate 3e-4 --mora_type 6 \
            --logging_steps 5  --use_bf16  --use_16bit --use_mora 
```

### pretraining

``` sh
deepspeed --num_gpus=8 --num_nodes=4 train.py \
        --micro_batch_size 16 --wandb_run_name mora-pretrain250m-r128 \
        --num_epochs 1 --wandb_project lora-pretrain --batch_size 1024 \
        --data_path <processed C4> --logging_steps 1 \
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
        --lora_r 128 --lora_alpha 64 --warmup_steps 1000  \
        --force_tqdm_update --lr_scheduler_type cosine \
        --max_steps 10000 --pretrain 250m \
        --train_embhead --learning_rate 5e-4 \
        --use_mora --use_relora --use_relora_step 2000  # ReMoRA merge per 2000 steps 
```

## Acknowledgement
Our Code is based on peft, alpaca-lora and ReLoRA
