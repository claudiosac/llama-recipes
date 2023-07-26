# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import torch
import os
import sys
import time
from typing import List
from tqdm import tqdm
from transformers import LlamaTokenizer, default_data_collator
from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model
from configs import fsdp_config, test_config
from utils.dataset_utils import get_preprocessed_dataset, generate_dataset_config
from utils.config_utils import update_config, generate_dataset_config
from rouge_score import rouge_scorer, scoring
from nltk.translate.bleu_score import sentence_bleu


def main(**kwargs):

    update_config((test_config, fsdp_config), **kwargs)

    if test_config.prompt_file is not None:
        assert os.path.exists(
            test_config.prompt_file
        ), f"Provided Prompt file does not exist {test_config.prompt_file}"
        with open(test_config.prompt_file, "r") as f:
            user_prompt = "\n".join(f.readlines())
    elif not sys.stdin.isatty():
        user_prompt = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(test_config.seed)
    torch.manual_seed(test_config.seed)
    
    model = load_model(test_config.model_name, test_config.quantization)
    tokenizer = LlamaTokenizer.from_pretrained(test_config.model_name)
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )

    if test_config.peft_model:
        model = load_peft_model(model, test_config.peft_model)

    dataset_config = generate_dataset_config(test_config, kwargs)

    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    print(f"--> Test Set Length = {len(dataset_test)}")

    # Create DataLoaders for the training and validation dataset

    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=test_config.val_batch_size,
        num_workers=test_config.num_workers_dataloader,
        pin_memory=True,
        sampler=None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    model.eval()

    test_preds_and_targets = list()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=False)
    bleu_result = {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}
    aggregator = scoring.BootstrapAggregator()

    start = time.perf_counter()

    for step, batch in enumerate(tqdm(test_dataloader, colour="red", desc=f"Test-set")):
        batch_targets = list()
        for key in batch.keys():
            batch_targets.append(batch[key]["target"])
            batch[key] = batch[key].to('cuda:0')

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
                length_penalty=test_config.length_penalty,
                **kwargs
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        for idx, target in enumerate(batch_targets):
            pred = output_text[idx]
            test_preds_and_targets.append((target, pred))
            score = scorer.score(target, pred)
            aggregator.add_scores(score)
            reference = [target.split()]
            candidate = pred.split()
            bleu_result["bleu1"] += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            bleu_result["bleu2"] += sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
            bleu_result["bleu3"] += sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
            bleu_result["bleu4"] += sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    e2e_inference_time = (time.perf_counter() - start) * 1000
    print(f"the inference time is {e2e_inference_time} ms")

    rouge_result = aggregator.aggregate()
    for key in rouge_result:
        rouge_result[key] = rouge_result[key].mid.fmeasure

    for key in bleu_result.keys():
        bleu_result[key] /= len(test_preds_and_targets)

    with open(test_config.summary_file, "w", encoding="utf-8") as f:
        f.write("ROUGE\tBLEU\n")
        f.write(str(rouge_result) + "\t" + str(bleu_result) + "\n")
        f.write("----\t-----\n")
        f.write("----\t-----\n")
        f.write("TARGET\tPREDICTION\n")
        for target, pred in test_preds_and_targets:
            f.write(target + "\t" + pred + "\n")


if __name__ == "__main__":
    fire.Fire(main)
