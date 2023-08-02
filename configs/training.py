# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config:
    model_name: str="/mnt/data/models/LLaMA2/7B_hf"
    enable_fsdp: bool= False 
    run_validation: bool=True
    batch_size_training: int=16
    num_epochs: int=3
    num_workers_dataloader: int=4
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=True
    mixed_precision: bool=True
    val_batch_size: int=3
    dataset = "qirphita_dataset"
    micro_batch_size: int=2
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=True
    output_dir: str = "/mnt/data/training_results/llama2-7b-8bit-qirphita"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = True
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    #
    checkpoint: str = "PATH/to/checkpoint"  # checkpoint must contain peft adapter e model.pkl
    log_interval: int = 10    # numero effetivo di steps di intervallo ->  (batch_size/micro_batch_size) * log_interval
    save_interval: int = 100  # numero effetivo di steps di intervallo ->  (batch_size/micro_batch_size) * save_interval
    eval_log_interval: int = 500  # intervallo effettivo di step per il log durante l'evaluation
    train_max_size: int = -1  # prende i primi N dati del dataset intero, per fare più veloce (-1 prende tutto)
    valid_max_size: int = -1  # prende i primi N dati del dataset intero, per fare più veloce (-1 prende tutto)
    #
    inference_interval: int = 300  # numero effetivo di steps di intervallo ->  (batch_size/micro_batch_size) * inference_interval (meglio che sia un multiplo di save_interval)
    inference_batch_size: int = 1  # > 1 not working, needs padding for inputs with different size in batch
    inference_max_size: int = 200  # prende i primi N dati del dataset intero, per fare più veloce (-1 prende tutto)
    inference_max_new_tokens: int = 256  # The maximum numbers of tokens to generate
    inference_do_sample: bool = True  # Whether or not to use sampling ; use greedy decoding otherwise.
    inference_min_length: int = None  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
    inference_use_cache: bool = True  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    inference_top_p: float = 1.0  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    inference_temperature: float = 1.0  # [optional] The value used to modulate the next token probabilities.
    inference_top_k: int = 50  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    inference_repetition_penalty: float = 1.0  # The parameter for repetition penalty. 1.0 means no penalty.
    inference_length_penalty: int = 1  # [optional] Exponential penalty to the length that is used with beam-based generation.
