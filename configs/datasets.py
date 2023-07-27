# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"


@dataclass
class qiansita_dataset:
    dataset: str = "qiansita_dataset"
    train_split: str = "train"
    test_split: str = "valid"
    inference_split: str = "test"
    max_words: int = 768
    data_path: str = "/mnt/data/qi_datasets/qiansita_data." + str(max_words) + ".json" if max_words > 0 else "/mnt/data/qi_datasets/qiansita_data.json"


@dataclass
class qisumita_dataset:
    dataset: str = "qisumita_dataset"
    train_split: str = "train"
    test_split: str = "valid"
    inference_split: str = "test"
    max_words: int = 1024
    data_path: str = "/mnt/data/qi_datasets/qisumita_data." + str(max_words) + ".json" if max_words > 0 else "/mnt/data/qi_datasets/qisumita_data.json"


@dataclass
class qiinstructita_dataset:
    dataset: str = "qiinstructita_dataset"
    train_split: str = "train"
    test_split: str = "valid"
    inference_split: str = "test"
    max_words: int = 512
    data_path: str = "/mnt/data/qi_datasets/qiinstructita_data." + str(max_words) + ".json" if max_words > 0 else "/mnt/data/qi_datasets/qiinstructita_data.json"