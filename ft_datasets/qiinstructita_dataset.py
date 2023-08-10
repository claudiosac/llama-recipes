# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import random

import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

PROMPT_DICT = {
    "prompt_input": (
        "Di seguito è riportata un'istruzione che descrive un compito da eseguire, insieme a un input che fornisce il contesto. "
        "Scrivi una risposta che completa in modo appropriato la richiesta.\n\n"
        "### Istruzione:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Di seguito è riportata un'istruzione che descrive un compito da eseguire. "
        "Scrivi una risposta che completa in modo appropriato la richiesta.\n\n"
        "### Istruzione:\n{instruction}\n\n### Response:"
    ),
}


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=512, max_size=-1, by_type=False, shuffle=False):
        print("Loading dataset: " + dataset_config.data_path + " (max_words=" + str(max_words) + ", split=" + partition + ", by_type=" + str(by_type) + ")")
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann["train"]
        elif partition == "valid":
            self.ann = self.ann["valid"]
        elif partition == "test":
            self.ann = self.ann["test"]

        types = ["ans", "qa", "sum", "ttl", "rph"]

        if by_type:
            data_by_type = {el: list for el in types}
            for el in self.ann:
                eltype = self.get_type(el)
                if eltype in types:
                    if len(data_by_type[eltype]) < max_size:
                        data_by_type[eltype].append(el)

                    stop = True
                    for t in data_by_type.keys():
                        if len(data_by_type[t]) < max_size:
                            stop = False
                            break

                    if stop:
                        break

            self.ann = list()
            for t in data_by_type.keys():
                for el in data_by_type[t]:
                    self.ann.append(el)

        else:
            if 0 < max_size < len(self.ann):
                self.ann = self.ann[0:max_size]

        if shuffle:
            random.shuffle(self.ann)

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask
        }

    def get_type(self, element):
        if "type" in element:
            return element["type"]
        else:
            instruction = element["instruction"]
            if instruction == "Genera un breve riassunto in italiano.":
                return "sum"
            elif instruction == "Riformula in italiano con tono neutro.":
                return "rph"
            elif instruction == "Genera un titolo in italiano.":
                return "ttl"
            elif "Rispondi a questa domanda in italiano:" in instruction:
                return "ans"
            elif "domanda e risposta in italiano" in instruction:
                return "qa"
            else:
                return instruction

    def get_batch(self, idx, batch_size=1):
        instructions, inputs, prompts, responses = list(), list(), list(), list()
        if idx >= len(self.ann):
            return None
        end = idx + batch_size if idx+batch_size <= len(self.ann) else len(self.ann)
        slice = self.ann[idx:end]
        for el in slice:
            input = el["input"]
            instruction = el["instruction"]
            prompt, response = "", el["output"]
            if el.get("input", "") == "":
                prompt = PROMPT_DICT["prompt_no_input"].format_map(el)
            else:
                prompt = PROMPT_DICT["prompt_input"].format_map(el)

            eltype = self.get_type(el)
            instructions.append(eltype)

            if eltype == "ans":
                inputs.append(instruction.replace("Rispondi a questa domanda in italiano:", "").strip())
            else:
                inputs.append(input)

            prompts.append(prompt)
            responses.append(response)
        return [inputs, prompts, responses, instructions]
