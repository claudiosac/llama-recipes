# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import copy

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import torch
import os
import sys
import time
from typing import List
from tqdm import tqdm
from transformers import LlamaTokenizer, default_data_collator
from inference.safety_utils import get_safety_checker
from inference.model_utils import load_model, load_peft_model
from configs import fsdp_config, test_config
from utils.dataset_utils import get_preprocessed_dataset
from utils.config_utils import update_config, generate_dataset_config
from rouge_score import rouge_scorer, scoring
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd


def main(**kwargs):

    update_config((test_config, fsdp_config), **kwargs)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(test_config.seed)
    torch.manual_seed(test_config.seed)
    
    model = load_model(test_config.model_name, test_config.quantization)
    tokenizer = LlamaTokenizer.from_pretrained(test_config.model_name)
    tokenizer.add_special_tokens({"pad_token": "<PAD>",})

    if test_config.peft_model is not None and test_config.peft_model != "":
        model = load_peft_model(model, test_config.peft_model)

    dataset_config = generate_dataset_config(test_config, kwargs)

    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
        max_size=200,
    )
    print(f"--> Test Set Length = {len(dataset_test)}")

    # Create DataLoaders for the training and validation dataset

    model.eval()

    xlsx_sheet1, xlsx_sheet2 = {"metric": list(), "value": list()}, {"instruction": list(), "input": list(), "target": list(), "prediction": list()}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=False)
    bleu_result = {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}
    aggregator = scoring.BootstrapAggregator()

    num_batches = int(len(dataset_test) / test_config.test_batch_size)
    for step in tqdm(range(0, num_batches), colour="red", desc="Test-set"):
        idx = step * test_config.test_batch_size
        [batch_inputs, batch_prompts, batch_targets, batch_instructions] = dataset_test.get_batch(idx, batch_size=test_config.test_batch_size)

        batch = tokenizer(batch_prompts, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}

        outputs = None
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=test_config.max_new_tokens,
                do_sample=test_config.do_sample,
                top_p=test_config.top_p,
                temperature=test_config.temperature,
                min_length=test_config.min_length,
                use_cache=test_config.use_cache,
                top_k=test_config.top_k,
                repetition_penalty=test_config.repetition_penalty,
                length_penalty=test_config.length_penalty
            )

        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if len(batch_targets) != len(batch_predictions):
            print("WARNING! Batch size different:", len(batch_targets), "!=", len(batch_predictions))
            continue

        for idx, target in enumerate(batch_targets):
            inp = batch_inputs[idx]
            ins = batch_instructions[idx]
            pred = batch_predictions[idx].split("### Response:")[1].strip()
            xlsx_sheet2["instruction"].append(ins)
            xlsx_sheet2["input"].append(inp)
            xlsx_sheet2["target"].append(target)
            xlsx_sheet2["prediction"].append(pred)
            score = scorer.score(target, pred)
            aggregator.add_scores(score)
            reference = [target.split(" ")]
            candidate = pred.split(" ")
            bleu_result["bleu1"] += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            bleu_result["bleu2"] += sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
            bleu_result["bleu3"] += sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
            bleu_result["bleu4"] += sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
            #print("P:", pred)
            #print("T:", target)
            #print(score, bleu_result)

    rouge_result = aggregator.aggregate()
    for key in rouge_result:
        rouge_result[key] = rouge_result[key].mid.fmeasure
        xlsx_sheet1["metric"].append(key)
        xlsx_sheet1["value"].append(rouge_result[key])

    for key in bleu_result.keys():
        bleu_result[key] /= len(xlsx_sheet2)
        xlsx_sheet1["metric"].append(key)
        xlsx_sheet1["value"].append(bleu_result[key])

    print(rouge_result)
    print(bleu_result)

    df1 = pd.DataFrame(xlsx_sheet1)
    df2 = pd.DataFrame(xlsx_sheet2)
    with pd.ExcelWriter(test_config.summary_file) as writer:
        df1.to_excel(writer, sheet_name="metrics", index=False)
        df2.to_excel(writer, sheet_name="text", index=False)


if __name__ == "__main__":
    fire.Fire(main)
