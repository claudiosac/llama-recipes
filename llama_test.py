# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import copy

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import numpy as np
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

    muse_model, bertscore, bertmodel = None, None, "bert-base-multilingual-uncased"
    bertscore = None
    if test_config.by_type:
        from evaluate import load
        import tensorflow as tf
        import tensorflow_hub as hub
        from sklearn.metrics.pairwise import cosine_similarity as cosine

        muse_model = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3", trainable=False)
        bertscore = load("bertscore")

    dataset_config = generate_dataset_config(test_config, kwargs)

    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
        max_size=test_config.max_size,
        by_type=test_config.by_type
    )
    print(f"--> Test Set Length = {len(dataset_test)}")

    # Create DataLoaders for the training and validation dataset

    model.eval()

    xlsx_sheet1, xlsx_sheet2 = {"type": list(), "metric": list(), "value": list()}, {"instruction": list(), "input": list(), "target": list(), "prediction": list()}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=False)
    aggregator = scoring.BootstrapAggregator()
    bleu_result = {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}
    bertscore_pairs = {"ans" : [list(), list()], "rph": [list(), list()]}

    stats_by_type = {
                     "ans": {"rouge": scoring.BootstrapAggregator(),
                             "bleu": {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}, "bertscore": 0, "acs": 0},
                     "sum":  {"rouge": scoring.BootstrapAggregator(),
                              "bleu": {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}},
                     "qa":  {"rouge": scoring.BootstrapAggregator(),
                             "bleu": {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}},
                     "ttl":  {"rouge": scoring.BootstrapAggregator(),
                              "bleu": {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}},
                     "rph":  {"rouge": scoring.BootstrapAggregator(),
                              "bleu": {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}, "bertscore": 0, "acs": 0},
                     } if test_config.by_type else None

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
            b1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            b2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
            b3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
            b4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

            bleu_result["bleu1"] += b1
            bleu_result["bleu2"] += b2
            bleu_result["bleu3"] += b3
            bleu_result["bleu4"] += b4

            if test_config.by_type:
                mode = ins.lower()
                stats_by_type[mode]["size"] += 1
                stats_by_type[mode]["rouge"].add_scores(score)
                stats_by_type[mode]["bleu"]["bleu1"] += b1
                stats_by_type[mode]["bleu"]["bleu2"] += b2
                stats_by_type[mode]["bleu"]["bleu3"] += b3
                stats_by_type[mode]["bleu"]["bleu4"] += b4

                if mode == "ans" or mode == "rph":
                    bertscore_pairs[mode][0].append(target)
                    bertscore_pairs[mode][1].append(pred)
                    seq_embeddings = muse_model([target, pred])
                    tf.reshape(seq_embeddings, [-1])
                    embeddings = seq_embeddings.numpy()
                    a = embeddings[0].reshape(1, -1)
                    b = embeddings[1].reshape(1, -1)
                    cosine_score = float(cosine(a, b)[0][0])
                    stats_by_type[mode]["acs"] += cosine_score

    rouge_result = aggregator.aggregate()
    for key in rouge_result:
        rouge_result[key] = rouge_result[key].mid.fmeasure
        xlsx_sheet1["type"].append("OVERALL")
        xlsx_sheet1["metric"].append(key)
        xlsx_sheet1["value"].append(rouge_result[key])

    for key in bleu_result.keys():
        bleu_result[key] /= len(xlsx_sheet2["input"])
        xlsx_sheet1["type"].append("OVERALL")
        xlsx_sheet1["metric"].append(key)
        xlsx_sheet1["value"].append(bleu_result[key])

    print("ROUGE:\n", rouge_result)
    print("BLEU:\n", bleu_result)

    if test_config.by_type and stats_by_type is not None:

        for mode in stats_by_type.keys():

            rouge_result_for_type = stats_by_type[mode]["rouge"].aggregate()
            for key in rouge_result_for_type:
                rouge_result[key] = rouge_result_for_type[key].mid.fmeasure
                xlsx_sheet1["type"].append(mode)
                xlsx_sheet1["metric"].append(key)
                xlsx_sheet1["value"].append(rouge_result_for_type[key])

            for key in stats_by_type[mode]["bleu"].keys():
                stats_by_type[mode]["bleu"][key] /= stats_by_type[mode]["size"]
                xlsx_sheet1["type"].append(mode)
                xlsx_sheet1["metric"].append(key)
                xlsx_sheet1["value"].append(stats_by_type[mode]["bleu"][key])

            if mode == "ans" or mode == "rph":
                bert_scores = bertscore.compute(predictions=bertscore_pairs[mode][1], references=bertscore_pairs[mode][0], model_type=bertmodel)

                bert_scores["precision"] = np.average(bert_scores["precision"])
                bert_scores["recall"] = np.average(bert_scores["recall"])
                bert_scores["f1"] = np.average(bert_scores["f1"])

                xlsx_sheet1["type"].append(mode)
                xlsx_sheet1["metric"].append("bertscore_PR")
                xlsx_sheet1["value"].append(bert_scores["precision"])

                xlsx_sheet1["type"].append(mode)
                xlsx_sheet1["metric"].append("bertscore_REC")
                xlsx_sheet1["value"].append(bert_scores["recall"])

                xlsx_sheet1["type"].append(mode)
                xlsx_sheet1["metric"].append("bertscore_F1")
                xlsx_sheet1["value"].append(bert_scores["f1"])

                stats_by_type[mode]["acs"] /= stats_by_type[mode]["size"]
                xlsx_sheet1["type"].append(mode)
                xlsx_sheet1["metric"].append("avg_cosine_similarity")
                xlsx_sheet1["value"].append(stats_by_type[mode]["acs"])

        print(stats_by_type)

    df1 = pd.DataFrame(xlsx_sheet1)
    df2 = pd.DataFrame(xlsx_sheet2)
    with pd.ExcelWriter(test_config.summary_file) as writer:
        df1.to_excel(writer, sheet_name="metrics", index=False)
        df2.to_excel(writer, sheet_name="text", index=False)


if __name__ == "__main__":
    fire.Fire(main)
