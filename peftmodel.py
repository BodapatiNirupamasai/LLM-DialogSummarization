import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from huggingface_hub import login
from config import  model_name, huggingface_dataset_name, output_dir, peft_model_path


def load_model_tokenizer_dataset(model_name,huggingface_dataset_name ):
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(huggingface_dataset_name)
    
    return original_model, tokenizer, dataset



def tokenize_data(dataset, tokenizer):

    def tokenize_function(example):
        start_prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary: '
        prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
        example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
        example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
        return example

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(['id', 'topic', 'dialogue', 'summary',])
    tokenized_datasets['validation'] = tokenized_datasets['validation'].remove_columns(['id', 'topic', 'dialogue', 'summary',])

    return tokenized_datasets


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def get_peft_trainer(peft_model, sample_tokenized_dataset):
    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3, 
        num_train_epochs=3,
        logging_steps=100,
    )
        
    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=sample_tokenized_dataset["train"],
    )

    return peft_trainer


def save_and_load_peft(peft_trainer):
    peft_trainer.model.save_pretrained(peft_model_path)
    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    peft_model = PeftModel.from_pretrained(peft_model_base, 
                                        peft_model_path, 
                                        torch_dtype=torch.bfloat16,
                                        is_trainable=False
                                        )
    return peft_model


def merge_model_and_push_to_huggingface():

    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained("merged_base_peft_model")
    token = 'Enter Token'
    login(token)
    merged_model.push_to_hub('nirupama1899/dialog-summarization')


def evaluate_model(peft_model, sample_tokenized_dataset, tokenizer):

    rouge = evaluate.load('rouge')

    peft_model_summaries = []
    dialogues = sample_tokenized_dataset['test'][:200]['dialogue']
    human_baseline_summaries = sample_tokenized_dataset['test'][:200]['summary']

    for idx, dialogue in enumerate(dialogues):
        prompt = f"""
        Summarize the following conversation

    {dialogue}

    Summary: """
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
        peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

        peft_model_summaries.append(peft_model_text_output)

    
    peft_model_results = rouge.compute(
        predictions= peft_model_summaries,
        references= human_baseline_summaries,
        use_aggregator=True,
        use_stemmer=True,
    )

    return peft_model_results


def main():

    original_model, tokenizer, dataset = load_model_tokenizer_dataset(model_name, huggingface_dataset_name)
    
    tokenized_datasets = tokenize_data(dataset, tokenizer)
    sample_tokenized_dataset = tokenized_datasets.filter(lambda example, index: index % 2 == 0, with_indices=True)
    
    lora_config = LoraConfig(
        r=32, 
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.025,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM 
    )

    peft_model = get_peft_model(original_model, lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))

    peft_trainer = get_peft_trainer(peft_model,sample_tokenized_dataset)
    peft_trainer.train()
    peft_model = save_and_load_peft(peft_trainer)
    
    # merge_model_and_push_to_huggingface(peft_model)
    peft_model_results = evaluate_model(peft_model, sample_tokenized_dataset, tokenizer)
    print(peft_model_results)


if __name__ == "__main__":
    main()