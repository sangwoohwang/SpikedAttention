import torch
from torch import nn
import numpy as np
import pandas as pd 
from utils_neuron_bert import *
# BERT
from transformers.models.bert.ma_bert.modeling_ma_bert import MA_BertForSequenceClassification
from transformers.models.bert.ma_bert.configuration_ma_bert import MA_BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification

# DistilBERT
# from transformers.models.distilbert.ma_distilbert.modeling_ma_distilbert import MA_DistilBertForSequenceClassification
# from transformers.models.distilbert.ma_distilbert.configuration_ma_distilbert import MA_DistilBertConfig
# from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification

from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric #Need to install using pip install datasets 
import datasets
import random
import argparse
from timm import utils

device = "cuda:0" if torch.cuda.is_available() else "cpu"
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def load_pretrained_model(num_labels, ckpt_file, model_checkpoint):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Change this if using different model
    config = MA_BertConfig(use_softmax_approx = False, 
                        share_softmax_nn=False,
                        softmax_input_size = 128,
                        softmax_hidden_layer_size = 128,
                        norm_type="Power", 
                        use_linformer=False, 
                        bert_encoder_hidden_act = "relu", 
                        verbose = True, 
                        num_labels = num_labels,
                        use_safetensors=True,
                        local_files_only=True,
                        )
                        
    # Change this if using different model
    student_model = MA_BertForSequenceClassification.from_pretrained(ckpt_file, 
                                                                           config=config)

    print(f"Done Generating State Dict from Pretrained ")
    return student_model

def preprocess_function(examples):
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key],  max_length=128, padding = "max_length", truncation=True,)
    return tokenizer(examples[sentence1_key], examples[sentence2_key],  max_length=128, padding = "max_length",truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    ### Classification - get the highest predicted logit
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    ### Regression
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

def show_random_elements(dataset, num_examples=10):
    print(dataset)
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    pd.set_option('display.width', 120)
    print(df)

def TotalLossForKD(student_output, teacher_output, alpha, temperature, beta, num_labels):
    soft_target_loss = nn.CrossEntropyLoss()(student_output['logits'].view(-1, num_labels)/temperature, 
                                            nn.Softmax(dim = -1)(teacher_output['logits'].view(-1, num_labels)/temperature))
    layer_loss = 0
    attn_loss = 0
    ### Change 12 to 6 if using distilroberta or distilbert
    for i in range(12):
        layer_loss += nn.MSELoss()(student_output['hidden_states'][i], teacher_output['hidden_states'][i])
        attn_loss += nn.MSELoss()(student_output['attentions'][i], teacher_output['attentions'][i])
    
    ### Change 12 to 6 if using distilroberta or distilbert
    layer_loss += nn.MSELoss()(student_output['hidden_states'][12], teacher_output['hidden_states'][12])
    
    return alpha * soft_target_loss \
        + (1 - alpha) * student_output.loss \
        + beta * (layer_loss + 100 * attn_loss)

def preprocess_logits_for_metrics(logits, labels):
    return logits[0]
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--pretrained_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--timestep', type=int, required=True)
    parser.add_argument('--base', type=float, required=True)

    
    args = parser.parse_args()
    print(args)
    seed = random.randint(0, 1000)
    if utils.is_primary(args):
        print("----Conversion of MA-BERT to Spiked Attention ----")
        print("Timestep:",args.timestep)
        print("Base:",args.base)

    ### Define task to fine tune bert on and to evaluate
    task = args.task

    ### Define what pre-trained model checkpoint you want to use 
    model_checkpoint = args.model

    ### Adjust the batch size to avoid out-of-memory errors
    batch_size = args.batch_size 


    print(f"Task Selected: {str.upper(task)}")

    ### For mnli-mm, the actual task is mnli. The rest remains the same
    actual_task = "mnli" if task == "mnli-mm" else task
    
    ### Loading data required for the GLUE task
    dataset = load_dataset("glue", actual_task)
    ### Loading the metric required for the GLUE task (e.g. Accuracy, F1 Score, MCC etc)
    metric = load_metric('glue', actual_task)
    
    ### Tokenizer to preprocess the input before feeding into the model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, model_max_length = 128)

    ### Preprocess all data loaded for the GLUE task
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

    ### Define your model here
    ### All tasks are single sentence or sentence pair classification, except STS-B, which is a regression task.
    ### Loading Pretrained Model   
    student_model = load_pretrained_model(num_labels, args.pretrained_file, model_checkpoint=model_checkpoint)

    
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        f"{args.save_dir}/{args.file_name}-{1797}-{task}", #Creates a directory named as provided
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy = "epoch",
        save_total_limit = 1,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        optim="adamw_torch",
        warmup_ratio = 0.1,
        do_train = False,
        do_predict = True,
        seed=seed
    )

    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    student_model, n_layer = replace_identity_by_module(student_model,0,batch_size)


    trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics = preprocess_logits_for_metrics

    )    
    if utils.is_primary(args):
        print("----Performance of Pre-trained MA-BERT----")
        print(trainer.evaluate())
        print(f"number of GFLOPs: {student_model.flops() / 1e9 :>7.3f}  ")
        print(f"Total ANN Energy (mJ): {student_model.flops_ANN().data / 1e9 :>7.3f}  ")

    model_2, n_layer = replace_ANN_by_SNN(student_model,n_layer=0,timestep=args.timestep,tau=args.base)

    

    SNN = Trainer(
    model=model_2,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics = preprocess_logits_for_metrics

    )
    if utils.is_primary(args):
        print("----Performance of SpikedAttention----")
        print(SNN.evaluate())
        print(f"Total SNN Energy (mJ): {student_model.flops_SNN().data / 1e9 :>7.3f}  ")
