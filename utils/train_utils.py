# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
import math
from typing import List
import shutil
import fire
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from torch.nn import functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.distributed.fsdp import StateDictType
import torch.distributed as dist
from pkg_resources import packaging
from .memory_utils import MemoryTrace
import model_checkpointing
import torch.cuda.nccl as nccl
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from policies import bfSixteen, fpSixteen,bfSixteen_mixed, get_llama_wrapper
from copy import deepcopy
from rouge_score import rouge_scorer, scoring
from nltk.translate.bleu_score import sentence_bleu

scaler = ShardedGradScaler()


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)


def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps,
          train_config, fsdp_config=None, local_rank=None, rank=None, wandb=None, resume_from=None, inference_dataset=None):
    """
    Trains the model on the given dataloader
    
    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons
    
    Returns: results dictionary containing average training and validation perplexity and loss
    """

    print(train_config.output_dir)

    # Create a gradient scaler for fp16
    scaler = torch.cuda.amp.GradScaler() if train_config.use_fp16 else None

    last_epoch, last_step, last_loss = 0, 0, 0
    if resume_from is not None and len(resume_from) == 4:
        last_epoch, last_step, last_loss, scaler_state = resume_from
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
            print("Scaler loaded from checkpoint : " + train_config.checkpoint)

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]
    results = {}
    best_val_loss = float("inf")
    for epoch in range(0, train_config.num_epochs):

        is_epoch_finished = True if (epoch == (last_epoch - 1) and last_step == len(train_dataloader)) else False
        continue_epoch = True if (epoch == (last_epoch - 1) and last_step < len(train_dataloader)) else False

        if epoch < (last_epoch - 1) or is_epoch_finished:
            print("Training Epoch " + str(epoch) + " SKIPPED!!")
            continue
        else:
            if continue_epoch:
                print("Continue Epoch " + str(epoch) + " from step " + str(last_step))
            else:
                print("Starting new epoch from 0")

        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = last_loss if continue_epoch else 0
            data_set_len = 0

            for step, batch in enumerate(tqdm(train_dataloader,colour="blue", desc=f"Training Epoch {epoch}")):

                if epoch == (last_epoch - 1) and step <= (last_step - 1):
                    continue

                for key in batch.keys():
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to('cuda')
                    # print(key, batch[key].device)
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                first_key = next(iter(batch))
                data_set_len += len(batch[first_key])
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        path_to_save = train_config.output_dir + "/last_checkpoint"
                        if os.path.exists(path_to_save):
                            shutil.rmtree(path_to_save)
                        model.save_pretrained(path_to_save)
                        torch.save({'epoch': epoch + 1, 'step': step + 1, 'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': total_loss / (step + 1), 'scheduler': lr_scheduler.state_dict(), 'scaler': scaler.state_dict()},
                                   path_to_save + "/checkpoint.pkl")

                        if wandb is not None and ((step + 1) % (gradient_accumulation_steps * train_config.log_interval) == 0 or step == len(train_dataloader) - 1):
                            loss_value = loss.detach().float().item()
                            num_step_for_current_epoch = ((step+1)-(last_step-1)) if continue_epoch else (step+1)
                            epochloss = total_loss.item() / num_step_for_current_epoch
                            wandb.log({"train/step_loss": round(loss_value, 6),
                                       "train/loss": round(epochloss, 6), "train/epoch": epoch+1,
                                       "train/step": (step+1)+(epoch*len(train_dataloader))})

                        if train_config.inference_interval > 0 and ((step + 1) % (gradient_accumulation_steps * train_config.inference_interval) == 0 or step == (len(train_dataloader) - 1)):
                            global_step = (step+1)+(epoch*len(train_dataloader))
                            inference(model, train_config, inference_dataset, tokenizer, global_step, wandb=wandb, by_type=train_config.inference_by_type)
                            model.train()
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()

                        path_to_save = train_config.output_dir + "/last_checkpoint"
                        if os.path.exists(path_to_save):
                            os.remove(path_to_save)
                        model.save_pretrained(save_path)
                        torch.save({'epoch': epoch + 1, 'step': step + 1, 'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': total_loss / (step + 1), 'scheduler': lr_scheduler.state_dict(), 'scaler': scaler.state_dict()},
                                   save_path + "/checkpoint.pkl")

                        if wandb is not None and ((step + 1) % (gradient_accumulation_steps * train_config.log_interval) == 0 or step == len(train_dataloader) - 1):
                            loss_value = loss.detach().float().item()
                            num_step_for_current_epoch = ((step + 1) - (last_step - 1)) if continue_epoch else (step + 1)
                            epochloss = total_loss.item() / num_step_for_current_epoch
                            wandb.log({"train/step_loss": round(loss_value, 6),
                                       "train/loss": round(epochloss, 6), "train/epoch": epoch+1,
                                       "train/step": (step+1)+(epoch*len(train_dataloader))})

                        if train_config.inference_interval > 0 and ((step + 1) % (gradient_accumulation_steps * train_config.inference_interval) == 0 or  step == (len(train_dataloader) - 1)):
                            global_step = (step+1)+(epoch*len(train_dataloader))
                            inference(model, train_config, inference_dataset, tokenizer, global_step, wandb=wandb, by_type=train_config.inference_by_type)
                            model.train()

                if (step + 1) % (gradient_accumulation_steps * train_config.save_interval) == 0 or step == len(train_dataloader) - 1:
                    save_path = train_config.output_dir + "/checkpoint_" + str(epoch+1) + "." + str(step+1) + "." + str((step+1)+(epoch*len(train_dataloader)))
                    model.save_pretrained(save_path)

                    #torch.save({'epoch': epoch+1, 'step': step+1, 'model_state_dict': model.state_dict()}, save_path + "/model.pkl")
                    torch.save({'epoch': epoch+1, 'step': step+1, 'optimizer_state_dict': optimizer.state_dict(),
                                'loss': total_loss/(step + 1),
                                'scheduler': lr_scheduler.state_dict(), 'scaler': scaler.state_dict()},
                               save_path + "/checkpoint.pkl")

        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        train_epoch_loss = total_loss / data_set_len if data_set_len > 0 else torch.tensor(total_loss, dtype=torch.float)
        train_perplexity = torch.exp(train_epoch_loss)
        
        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        
        print(f"Max CUDA memory allocated was {memtrace.peak} GB")
        print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
        print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
        print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

        clear_gpu_cache()
            
        # Update the learning rate as needed
        lr_scheduler.step()
          
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, rank, tokenizer, epoch, wandb=wandb)
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                
                if train_config.use_peft:
                    model.save_pretrained(train_config.output_dir)   
                    print(f"PEFT modules are saved in {train_config.output_dir} directory")
                    
                else:
                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        model_checkpointing.save_model_checkpoint(model, optimizer, rank, train_config, epoch=1)
                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                        print(" we are about to save the models *******")
                        
                        model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config)
                        if train_config.save_optimizer:
                            model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)

                    if not train_config.use_peft and train_config.save_optimizer:
                        model_checkpointing.save_optimizer_checkpoint(model, optimizer, rank, train_config, epoch=1)

            if local_rank == 0 and eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
            val_loss.append(best_val_loss)
            val_prep.append(eval_ppl)

        print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}")
        lr_scheduler.step()

    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep) 
        avg_eval_loss = sum(val_loss)/len(val_loss) 

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss

    return results


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, epoch, wandb=None):
    """
    Evaluates the model on the given dataloader
    
    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions
    
    Returns: eval_ppl, eval_epoch_loss
    """
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    eval_dataset_len = 0
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc=f"Training Epoch {epoch}")):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                first_key = next(iter(batch))
                eval_dataset_len += len(batch[first_key])
                
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )

            if wandb is not None and step > 0 and (step % train_config.eval_log_interval == 0 or step == len(eval_dataloader) - 1):
                loss_value = loss.detach().float().item()
                perplexity_value = eval_loss.item() / (train_config.eval_log_interval * step)
                wandb.log({"eval/step_loss": round(loss_value, 6), "eval/loss": round(perplexity_value, 6), "eval/epoch": epoch + 1,
                           "eval/step": (step+1)+(epoch*len(eval_dataloader)), "eval/perplexity": math.exp(perplexity_value)})
    
    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    
    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / eval_dataset_len
    eval_ppl = torch.exp(eval_epoch_loss)
    
    # Print evaluation metrics
    print(f" {eval_ppl=} {eval_epoch_loss=}")
    return eval_ppl, eval_epoch_loss


def predict(model, tokenizer, batch_prompts, train_config):
    batch = tokenizer(batch_prompts, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}

    outputs = None
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=train_config.inference_max_new_tokens,
            do_sample=train_config.inference_do_sample,
            top_p=train_config.inference_top_p,
            temperature=train_config.inference_temperature,
            min_length=train_config.inference_min_length,
            use_cache=train_config.inference_use_cache,
            top_k=train_config.inference_top_k,
            repetition_penalty=train_config.inference_repetition_penalty,
            length_penalty=train_config.inference_length_penalty
        )

    batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return batch_predictions


def inference(model, train_config, dataset, tokenizer, global_step, wandb=None, by_type=False):

    model.eval()

    test_preds_and_targets = list()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=False)
    bleu_result = {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}
    aggregator = scoring.BootstrapAggregator()
    num_batches = int(len(dataset) / train_config.inference_batch_size)

    stats_by_type = {
        "ans": {"rouge": scoring.BootstrapAggregator(),
                "bleu": {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}, "size": 0},
        "sum": {"rouge": scoring.BootstrapAggregator(),
                "bleu": {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}, "size": 0},
        "qa": {"rouge": scoring.BootstrapAggregator(),
               "bleu": {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}, "size": 0},
        "ttl": {"rouge": scoring.BootstrapAggregator(),
                "bleu": {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}, "size": 0},
        "rph": {"rouge": scoring.BootstrapAggregator(),
                "bleu": {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}, "size": 0},
    } if by_type else None

    data_log = dict()

    for step in tqdm(range(0, num_batches), colour="red", desc="Inference"):
        idx = step * train_config.inference_batch_size
        [batch_inputs, batch_prompts, batch_targets, batch_instructions] = dataset.get_batch(idx, batch_size=train_config.inference_batch_size)

        batch_predictions = predict(model, tokenizer, batch_prompts, train_config)
        if len(batch_targets) != len(batch_predictions):
            print("WARNING! Batch size different:", len(batch_targets), "!=", len(batch_predictions))
            continue

        for idx, target in enumerate(batch_targets):
            ins = batch_instructions[idx]
            inp = batch_inputs[idx]
            pred = batch_predictions[idx].split("### Response:")[1].strip()
            test_preds_and_targets.append((inp, target, pred))
            score = scorer.score(target, pred)
            aggregator.add_scores(score)
            reference = [target.split(" ")]
            candidate = pred.split(" ")
            bleu_result["bleu1"] += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            bleu_result["bleu2"] += sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
            bleu_result["bleu3"] += sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
            bleu_result["bleu4"] += sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

            if by_type:
                reference = [target.split(" ")]
                candidate = pred.split(" ")
                mode = ins.lower()
                stats_by_type[mode]["size"] += 1
                rouge_score_for_type = scorer.score(target, pred)
                stats_by_type[mode]["rouge"].add_scores(rouge_score_for_type)
                stats_by_type[mode]["bleu"]["bleu1"] += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
                stats_by_type[mode]["bleu"]["bleu2"] += sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
                stats_by_type[mode]["bleu"]["bleu3"] += sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
                stats_by_type[mode]["bleu"]["bleu4"] += sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    rouge_result = aggregator.aggregate()
    for key in rouge_result:
        rouge_result[key] = rouge_result[key].mid.fmeasure

    for key in bleu_result.keys():
        bleu_result[key] /= len(test_preds_and_targets)

    if wandb is not None:
        data_log.update({"inference/rouge1": rouge_result["rouge1"], "inference/rouge2": rouge_result["rouge2"],
                         "inference/rougeL": rouge_result["rougeL"], "inference/rougeLsum": rouge_result["rougeLsum"],
                         "inference/bleu1": bleu_result["bleu1"], "inference/bleu2": bleu_result["bleu2"],
                         "inference/bleu3": bleu_result["bleu3"], "inference/bleu4": bleu_result["bleu4"], "inference/step": global_step})

    print("ROUGE:\n", rouge_result)
    print("BLEU:\n", bleu_result)

    if by_type and stats_by_type is not None:

        for mode in stats_by_type.keys():
            if stats_by_type[mode]["size"] > 0:
                rouge_result_for_type = stats_by_type[mode]["rouge"].aggregate()
                for key in rouge_result:
                    rouge_result_for_type[key] = rouge_result_for_type[key].mid.fmeasure
                for key in stats_by_type[mode]["bleu"].keys():
                    stats_by_type[mode]["bleu"][key] /= stats_by_type[mode]["size"]

                print(mode, ": ROUGE :", rouge_result_for_type)
                print(mode, ": BLEU :", stats_by_type[mode]["bleu"])

                if wandb is not None:
                    data_log.update({"inference/" + mode + "_rouge1": rouge_result["rouge1"], "inference/" + mode + "_rouge2": rouge_result["rouge2"],
                                     "inference/" + mode + "_rougeL": rouge_result["rougeL"], "inference/" + mode + "_rougeLsum": rouge_result["rougeLsum"]})

    if wandb is not None:
        wandb.log(data_log)


def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")
                
                
def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
            torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and packaging.version.parse(torch.version.cuda).release >= (11, 0)
            and dist.is_nccl_available()
            and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy
