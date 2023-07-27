from dataclasses import dataclass
from typing import ClassVar

@dataclass
class test_config:
    model_name: str = "PATH/to/LLAMA/7B"
    dataset:str = "qiansita_dataset"
    test_batch_size: int = 1   # > 1 not working, needs padding for inputs with different size in batch
    num_workers_dataloader: int = 4
    peft_model: str = ""  #  --peft_model /mnt/data/training_results/llama2-7b-8bit-qiansita_base/checkpoint_2.40718
    quantization: bool = True
    summary_file: str = "PATH/to/summary/file"
    seed: int = 42  # seed value for reproducibility
    max_new_tokens: int = 256  # The maximum numbers of tokens to generate
    do_sample: bool = True  # Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int = None  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool = True  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float = 1.0  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float = 1.0  # [optional] The value used to modulate the next token probabilities.
    top_k: int = 50  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float = 1.0  # The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int = 1  # [optional] Exponential penalty to the length that is used with beam-based generation.
