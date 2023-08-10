# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from functools import partial

from ft_datasets import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
    get_qiansita_dataset,
    get_qisumita_dataset,
    get_qiinstructita_dataset,
    get_qirphita_dataset,
    get_qittlita_dataset
)
from typing import Optional


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=224),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "qiansita_dataset": partial(get_qiansita_dataset),
    "qisumita_dataset": partial(get_qisumita_dataset),
    "qiinstructita_dataset": partial(get_qiinstructita_dataset),
    "qirphita_dataset": partial(get_qirphita_dataset),
    "qittlita_dataset": partial(get_qittlita_dataset)
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train", max_size: int = -1
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        if split == "train":
            return dataset_config.train_split
        elif split == "test":
            return dataset_config.test_split
        elif split == "inference":
            return dataset_config.inference_split
        return dataset_config.train_split
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
        max_words=dataset_config.max_words,
        max_size=max_size,
    )
