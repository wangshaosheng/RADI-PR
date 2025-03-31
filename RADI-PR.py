from tqdm import tqdm
import torch.optim as optim
import csv
import json
import numpy
import faiss
import pickle
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import sys
import os
import time
import pytorch_lightning as pl
import torch
import pathlib
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, MeanMetric
from transformers import T5TokenizerFast, T5ForConditionalGeneration, RobertaModel, RobertaConfig, GPT2TokenizerFast, \
    RobertaTokenizerFast, GPTNeoForCausalLM
from torch.nn import Parameter
import aprkits.nn.functional as f
from aprkits.callbacks import AutoEpochEndCallbackForLossAccFullAcc, \
    AutoBatchEndForLM, BestModelCheckpoint, StatefulModelCheckpoint
from aprkits.data import BatchEncodingDataset
from aprkits.data.datasets import DataPreprocessor 

from aprkits.nn import CatGraphNet
from aprkits.optim import CosineWarmupScheduler
from aprkits.tokenizers import NumberTokenizer
from aprkits.utils import set_trainer_epoch, load_model_or_checkpoint

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from collections import deque
from peft import (get_peft_config,
                  get_peft_model,
                  LoraConfig,
                  TaskType,
                  PromptEncoderConfig,
                  AutoPeftModelForCausalLM,
                  PromptTuningConfig,
                  PromptTuningInit,
                  PrefixTuningConfig,
                  IA3Config)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score, AUROC

class Args(Namespace):
    max_epochs: int
    warmup_steps: int
    batch_size: int
    learning_rate: float
    use_lr_scheduler: bool
    optim_eps: float
    dropout_rate: float
    accelerator: str
    devices: List[int]
    torch_num_threads: int
    model_max_length: int

    early_stop_monitor: str = 'v.loss'
    early_stop_min_delta: float = 0.1
    early_stop_mode: str = 'min'
    early_stop_patience: int = 8

    limit_train_batches: Optional[int]
    limit_val_batches: Optional[int]
    limit_test_batches: Optional[int]
    limit_pred_batches: Optional[int]

    data_input_dir: str
    tokenizer_cache_dir: Optional[str]
    model_cache_dir: Optional[str]
    summary_dir: str
    ckpt_dir: str
    model_dir: str
    no_model_save: bool
    save_top_k: int

    experiment: str
    representation: str


TIME_STAMP = str(datetime.now().timestamp())

arg_parser = ArgumentParser(
    'Model Trainer Script',
    description='This script is for running different configurations of model fitting.')

arg_parser.add_argument(
    '-i', '--data_input_dir', type=str, required=True,
    help='Specify the data for training.'
         'Input folder should contain .train, .valid and .test files of corresponding data.'
)
arg_parser.add_argument(
    '-X', '--experiment', type=str, choices=['t5', 'codet5'], required=True,
    help='Experiment configuration.'
         'Option t5 defines the experiments on t5-base pretrained model, '
         'while codet5 will use Salesforce/codet5-base. '
         'Graph is based on a composite model described in the paper.')
arg_parser.add_argument(
    '-r', '--representation', type=str, choices=['text'], required=True,
    help='Data representation that will be used during training.')
arg_parser.add_argument(
    '-T', '--task', default='ILPR', type=str, choices=['ILPR', 'CLPR'], required=False,
    help='Task configuration. '
         'Option ILPR defines the task for Intra-Language Program Repair, '
         'while CLPR refers to Continual Cross-Language Program Repair.')


arg_parser.add_argument(
    '-E', '--max_epochs', default=50, type=int, required=False,
    help='Maximum number of epochs the model can train. '
         'Early stopping might kill training before this. '
         'Affects learning rate, if a learning rate scheduler is used (pass -ls or --use_lr_scheduler).')
arg_parser.add_argument(
    '-w', '--warmup_steps', default=1, type=int, required=False,
    help='Specifies warmup steps measured in epochs, until specified learning rate is reached.')
arg_parser.add_argument(
    '-b', '--batch_size', default=8, type=int, required=False,
    help='Specifies the batch size used during forward pass.')
arg_parser.add_argument(
    '-lr', '--learning_rate', default=5e-5, type=float, required=False,
    help='Specify a learning rate. Is a scheduler is used, then this will be the peak of your lr.')
arg_parser.add_argument(
    '-ls', '--use_lr_scheduler', action='store_true', help='Whether to use (linear) learning rate scheduler or not.')
arg_parser.add_argument(
    '-e', '--optim_eps', default=1e-8, type=float, required=False,
    help='Epsilon passed to the optimizer.')
arg_parser.add_argument(
    '-d', '--dropout_rate', default=0.2, type=float, required=False,
    help='Dropout rate used during training.')
arg_parser.add_argument(
    '-a', '--accelerator', default='gpu', choices=['cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'auto'], required=False,
    help='The type of accelerator your model will be running on.')
arg_parser.add_argument(
    '-D', '--devices', nargs='+', default=[0], type=int, required=False,
    help='Visible devices.')
arg_parser.add_argument(
    '-t', '--torch_num_threads', default=6, type=int, required=False,
    help='Number of threads allowed to be used by torch.')
arg_parser.add_argument(
    '-ml', '--model_max_length', default=512, type=int, required=False,
    help='Maximum sequence length of your model. Tokenization will use this number, too.')

arg_parser.add_argument(
    '-em', '--early_stop_monitor', default='v.loss', type=str, choices=['v.loss', 'v.acc', 'v.full'], required=False,
    help='Metric to monitor during early stopping.')
arg_parser.add_argument(
    '-ed', '--early_stop_min_delta', default=0.05, type=float, required=False,
    help='Minimum change required on watched metric, during patience time.')
arg_parser.add_argument(
    '-eM', '--early_stop_mode', default='min', type=str, choices=['min', 'max'], required=False,
    help='Monitor mode of watched metric.')
arg_parser.add_argument(
    '-ep', '--early_stop_patience', default=8, type=int, required=False,
    help='Patience interval, measured in epochs.')

arg_parser.add_argument(
    '-ltb', '--limit_train_batches', default=None, type=int, required=False,
    help='Set, if you want to limit training batches. Useful for brainstorming.')
arg_parser.add_argument(
    '-lvb', '--limit_val_batches', default=None, type=int, required=False,
    help='Set, if you want to limit validation batches. Useful for brainstorming.')
arg_parser.add_argument(
    '-lTb', '--limit_test_batches', default=None, type=int, required=False,
    help='Set, if you want to limit test batches.')
arg_parser.add_argument(
    '-lpb', '--limit_pred_batches', default=None, type=int, required=False,
    help='Set, if you want to limit prediction batches. Useful when you want to see some quick examples.')

arg_parser.add_argument(
    '-tcD', '--tokenizer_cache_dir', default=None, required=False,
    help='Specifies cache dir that will be used by transformers, when downloading from hub.')
arg_parser.add_argument(
    '-mcD', '--model_cache_dir', default=None, required=False,
    help='Specifies cache dir that will be used by transformers, when downloading from hub.')
arg_parser.add_argument(
    '-sD', '--summary_dir', default=f'data/summary/{TIME_STAMP}', required=False,
    help='Specifies where the summaries should be written.')
arg_parser.add_argument(
    '-pD', '--preds_dir', default=f'data/preds/{TIME_STAMP}', required=False,
    help='Specifies output directory for predictions.')
arg_parser.add_argument(
    '-xD', '--ckpt_dir', default=f'data/checkpoints/{TIME_STAMP}', required=False,
    help='Specifies directory for storing model checkpoints.')
arg_parser.add_argument(
    '-mD', '--model_dir', default=f'data/models/{TIME_STAMP}', required=False,
    help='Specifies directory to store your trained model.')
arg_parser.add_argument(
    '-nS', '--no_model_save', action='store_true',
    help='Pass if you do not want to save the trained model. Checkpoints will still be stored.')

arg_parser.add_argument(
    '-st', '--save_top_k', default=10, type=int, required=False,
    help='Maximum number of checkpoints to be saved. '
         'Advised to be larger than, or equal to patience.')



class CLPRLoss(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, lambda_4=1.0, ignore_index=-100, reduction='mean'):
        super(CLPRLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Define CrossEntropyLoss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def source_cross_entropy_loss(self, logits, targets):
        """
        Calculate the cross-entropy loss for the source language
        """
        return self.ce_loss(logits.transpose(1, 2).reshape(-1, logits.size(1)), targets.view(-1))

    def domain_adaptation_loss(self, logits, targets):
        """
        Assume logits are the outputs from the source domain, and targets are labels for the target domain.
        Calculate the adaptation loss between the source and target domains.
        """
        return self.ce_loss(logits.transpose(1, 2).reshape(-1, logits.size(1)), targets.view(-1))

    def target_entropy_loss(self, target_prob):
        """
        Calculate the entropy loss for the target language, encouraging the model to be more confident in its predictions for the target language.
        """
        entropy_loss = -(target_prob * target_prob.log()).sum(dim=-1).mean()
        return entropy_loss

    def calculate_confidence(self, logits):
        """
        Calculate confidence: apply softmax to each token and select the maximum probability value.
        """
        probabilities = F.softmax(logits, dim=-1)
        confidence, _ = torch.max(probabilities, dim=-1)
        return confidence

    def confidence_loss(self, confidence, threshold=0.7):
        """
        Calculate confidence loss, penalizing confidence values below the threshold.
        """
        return F.relu(threshold - confidence).mean()

    def forward(self, logits, targets, target_prob=None):
        """
        Calculate the final weighted loss
        """
        # 1. Calculate source language cross-entropy loss
        source_loss = self.source_cross_entropy_loss(logits, targets)

        # 2. Calculate domain adaptation loss
        domain_loss = 0  # Assuming no domain adaptation loss

        # 3. Calculate target language entropy loss (if target language probability distribution is provided)
        target_loss = 0
        if target_prob is not None:
            target_loss = self.target_entropy_loss(target_prob)

        # 4. Calculate confidence loss
        confidence = self.calculate_confidence(logits)
        confidence_loss_value = self.confidence_loss(confidence)

        # Calculate total loss
        total_loss = (self.lambda_1 * source_loss + 
                      self.lambda_2 * domain_loss + 
                      self.lambda_3 * target_loss + 
                      self.lambda_4 * confidence_loss_value)

        return total_loss


class LitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()


        self.max_epochs = prog_args.max_epochs
        self.warmup = prog_args.warmup_steps
        self.lr = prog_args.learning_rate
        self.optim_eps = prog_args.optim_eps
        self.dropout = prog_args.dropout_rate

        # Initialize confidence
        self.confidence_threshold = Parameter(torch.tensor(0.5, requires_grad=True))

        if prog_args.experiment == 't5':
            self.tokenizer = T5TokenizerFast.from_pretrained(
                'model_t5/t5-base', cache_dir=prog_args.tokenizer_cache_dir, ignore_mismatched_sizes=True)
            self.model = T5ForConditionalGeneration.from_pretrained(
                'model_t5/t5-base', cache_dir=Path(prog_args.model_cache_dir).joinpath('t5-base'), ignore_mismatched_sizes=True)

        elif prog_args.experiment == 'codet5':
            self.tokenizer = RobertaTokenizerFast.from_pretrained(
                'Salesforce/codet5-base', cache_dir=prog_args.tokenizer_cache_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(
                'Salesforce/codet5-base', cache_dir=Path(prog_args.model_cache_dir).joinpath('Salesforce/codet5-base'))

        else:
            raise NotImplementedError(f'Sorry, experiments on {prog_args.experiment} is not implemented.')



        # self.criterion = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        if prog_args.task == 'CLPR':
            self.criterion = CLPRLoss()  
        else:
            self.criterion = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)  

        self.train_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, mdmc_average='global')   # , top_k=2
        self.val_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, mdmc_average='global')
        self.test_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, mdmc_average='global')

        self.train_full_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, subset_accuracy=True)
        self.val_full_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, subset_accuracy=True)
        self.test_full_accuracy = Accuracy(ignore_index=self.tokenizer.pad_token_id, subset_accuracy=True)

        self.train_loss_metric = MeanMetric(nan_strategy='ignore')
        self.val_loss_metric = MeanMetric(nan_strategy='ignore')
        self.test_loss_metric = MeanMetric(nan_strategy='ignore')
        self.optimizer_param_groups = [{}]

        self._test_pred_labels = torch.tensor([], dtype=torch.int32, device='cpu')
        self._test_true_labels = torch.tensor([], dtype=torch.int32, device='cpu')

        num_classes = self.model.config.vocab_size
        self.train_precision = Precision(ignore_index=self.tokenizer.pad_token_id, average='macro', num_classes=num_classes, mdmc_reduce="global")
        self.val_precision = Precision(ignore_index=self.tokenizer.pad_token_id, average='macro', num_classes=num_classes, mdmc_reduce="global")
        self.test_precision = Precision(ignore_index=self.tokenizer.pad_token_id, average='macro', num_classes=num_classes, mdmc_reduce="global")

        self.train_recall = Recall(ignore_index=self.tokenizer.pad_token_id, average='macro', num_classes=num_classes, mdmc_reduce="global")
        self.val_recall = Recall(ignore_index=self.tokenizer.pad_token_id, average='macro', num_classes=num_classes, mdmc_reduce="global")
        self.test_recall = Recall(ignore_index=self.tokenizer.pad_token_id, average='macro', num_classes=num_classes, mdmc_reduce="global")

        self.train_f1 = F1Score(ignore_index=self.tokenizer.pad_token_id, average='macro', num_classes=num_classes, mdmc_reduce="global")
        self.val_f1 = F1Score(ignore_index=self.tokenizer.pad_token_id, average='macro', num_classes=num_classes, mdmc_reduce="global")
        self.test_f1 = F1Score(ignore_index=self.tokenizer.pad_token_id, average='macro', num_classes=num_classes, mdmc_reduce="global")
        
        self.train_auc = AUROC(num_classes=num_classes, average='macro', multi_class='ovr')
        self.val_auc = AUROC(num_classes=num_classes, average='macro', multi_class='ovr')
        self.test_auc = AUROC(num_classes=num_classes, average='macro', multi_class='ovr')

        # Load vectorized change information
        with open('vectorized_intervention_strategies.json', 'r') as f:
            self.vectorized_strategies = json.load(f)

    def calculate_confidence(self, logits):
        # Use softmax to calculate the confidence of the output
        probabilities = F.softmax(logits, dim=-1)
        confidence, _ = torch.max(probabilities, dim=-1)
        return confidence

    def retrieve_info(self, src_text):
        """ Get the most relevant matches from vectorized_intervention_strategies.json """
        retrieved_info = []

        for key in self.vectorized_strategies:
            if key in src_text:  
                retrieved_info.append(self.vectorized_strategies[key])

        if not retrieved_info:
            # print(f"[WARNING] No retrieved info found for: {src_text}")  # Debug
            return None

        input_ids = [info['input_ids'][0] if isinstance(info['input_ids'], list) and len(info['input_ids']) > 0 else [] for info in retrieved_info]
        attention_masks = [info['attention_mask'][0] if isinstance(info['attention_mask'], list) and len(info['attention_mask']) > 0 else [] for info in retrieved_info]
        labels = [info['labels'][0] if isinstance(info['labels'], list) and len(info['labels']) > 0 else [] for info in retrieved_info]
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long) if input_ids else torch.empty((0,), dtype=torch.long)
        attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long) if attention_masks else torch.empty((0,), dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long) if labels else torch.empty((0,), dtype=torch.long)

        return input_ids_tensor, attention_masks_tensor, labels_tensor



    def _save_test_preds(self, pred_tensor):
        labels = pred_tensor.argmax(dim=-1).flatten().to(torch.int32)
        self._test_pred_labels = torch.concat((self._test_pred_labels, labels))

    def forward(self, *args, **kwargs):
        y = self.model(*args, **kwargs)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=self.optim_eps)

        optim_config = {
            'optimizer': optimizer
        }

        if prog_args.use_lr_scheduler:
            scheduler = CosineWarmupScheduler(
                optimizer=optimizer, warmup_iters=self.warmup, total_iters=self.max_epochs)
            optim_config['lr_scheduler'] = scheduler

        self.optimizer_param_groups = optimizer.param_groups

        return optim_config



    def forward_batch(self, batch, step_type: str = None):
        if prog_args.experiment == 't5' or prog_args.experiment == 'codet5':
            src_data, tgt_data, src_data_mask, _ = batch

            # Retrieving change information
            retrieved_embeddings = []
            for src in src_data:
                src_text = self.tokenizer.decode(src)
                retrieved_info = self.retrieve_info(src_text)
                if retrieved_info:
                    input_ids, attention_masks, labels = retrieved_info
                    embeddings = self.model.shared(input_ids.to(self.device))
                    retrieved_embeddings.append((embeddings, labels))

            model_output = self(
                input_ids=src_data,
                labels=tgt_data,
                attention_mask=src_data_mask
            )
            logits = model_output.logits  

            # Calculating model confidence
            confidence = self.calculate_confidence(logits).detach() 

            # Convert confidence tensor to a scalar (mean in this case)
            confidence_value = confidence.mean()  

            for i in range(labels.size(1)):
                label = labels[0, i]
                if label != self.tokenizer.pad_token_id:
                    if confidence_value < self.confidence_threshold:
                        if embeddings.shape[-1] != logits.shape[-1]:
                            embeddings = torch.nn.Linear(embeddings.shape[-1], logits.shape[-1]).to(embeddings.device)(embeddings)

                        logits[:, i, :embeddings.shape[1]] = embeddings[i, :]
                    logits[:, i, :] *= confidence_value  

            logits = logits.transpose(-1, -2)

            # Adjust loss based on confidence
            loss = self.criterion(logits, tgt_data)
            if confidence_value < self.confidence_threshold:
                loss *= 1.5  # Increase the loss for low confidence samples
            else:
                loss *= 0.5  # Decrease the loss for high confidence samples

            loss_val = self.train_loss_metric(loss)
            acc_val = self.train_accuracy(logits, tgt_data)

            
            logits = logits.transpose(-1, -2)  
            preds = torch.argmax(logits, dim=-1) 
            tgt_data = tgt_data.long()
            assert preds.shape == tgt_data.shape, f"Shape mismatch: preds {preds.shape}, tgt_data {tgt_data.shape}"
            # ** Precision, Recall, F1**
            precision_val = self.train_precision(preds, tgt_data)
            recall_val = self.train_recall(preds, tgt_data)
            f1_val = self.train_f1(preds, tgt_data)
    
            self.log("train_loss", loss_val, prog_bar=True)
            self.log("train_accuracy", acc_val, prog_bar=True)
            self.log("train_precision", precision_val, prog_bar=True)
            self.log("train_recall", recall_val, prog_bar=True)
            self.log("train_f1", f1_val, prog_bar=True)
            logits = model_output.logits
            logits = logits.transpose(-1, -2)

            best_perplexity = float('inf')  
            best_cosine_similarity = -float('inf')  
            best_mse_score = float('inf')  

            perplexity = torch.exp(loss / 8)  
            if perplexity.item() < best_perplexity:
                best_perplexity = perplexity.item()
            logits_probs = F.softmax(logits, dim=1)  # [batch_size, vocab_size, sequence_length]
            predicted_indices = torch.argmax(logits_probs, dim=1)  # [batch_size, sequence_length]
            tgt_data = torch.clamp(tgt_data, max=logits.size(1) - 1)  
            tgt_one_hot = F.one_hot(tgt_data, num_classes=logits.size(1)).float() 
            mask = tgt_data != self.tokenizer.pad_token_id  
            pred_one_hot = F.one_hot(predicted_indices, num_classes=logits.size(1)).float()
            pred_one_hot_flat = pred_one_hot.view(-1, pred_one_hot.size(2))  # [batch_size * sequence_length, vocab_size]
            tgt_one_hot_flat = tgt_one_hot.view(-1, tgt_one_hot.size(2))  # [batch_size * sequence_length, vocab_size]
            pred_one_hot_flat = pred_one_hot_flat[mask.view(-1)]  
            tgt_one_hot_flat = tgt_one_hot_flat[mask.view(-1)]  
            cosine_similarity = F.cosine_similarity(pred_one_hot_flat, tgt_one_hot_flat, dim=-1)
            similarity_score = cosine_similarity.mean()
            if similarity_score.item() > best_cosine_similarity:
                best_cosine_similarity = similarity_score.item()
            mse_score = F.mse_loss(pred_one_hot_flat, tgt_one_hot_flat)
            if mse_score.item() < best_mse_score:
                best_mse_score = mse_score.item()
            self.log("Perplexity: ", best_perplexity, prog_bar=True)
            self.log("Cosine Similarity:", best_cosine_similarity, prog_bar=True)
            self.log("MSE Score:", best_mse_score, prog_bar=True)

            logits = f.lift_predictions(logits, tgt_data, ignore_index=self.tokenizer.pad_token_id)
            full_acc_val = self.train_full_accuracy(logits, tgt_data)


            return {
                'logits': model_output.logits,
                'labels': tgt_data,
                'loss': loss,
                'mean_loss': loss_val,
                'accuracy': acc_val,
                'precision': precision_val,  
                'recall': recall_val, 
                'f1': f1_val,  
                'full_accuracy': full_acc_val,
                'step_type': step_type
            }
        else:
            raise NotImplementedError()


    def training_step(self, train_batch, batch_idx, *args, **kwargs):
        out = self.forward_batch(train_batch, 'train')
        return out

    def validation_step(self, val_batch, batch_idx, *args, **kwargs):
        out = self.forward_batch(val_batch, 'validation')
        return out

    def test_step(self, test_batch, batch_idx, *args, **kwargs):
        out = self.forward_batch(test_batch, 'test')
        return out

    

    def _get_loader(self, dt: str):
        if prog_args.representation == 'text':
            prefx = 'text'
        elif prog_args.representation == 'cmdseqtoken':
            prefx = 'cmdseq'
        elif prog_args.representation == 'graphtext':
            prefx = 'graph+sequence'
        else:
            raise NotImplementedError()

        shuffle = False
        if dt == 'train':
            shuffle = True

        postfix = ''
        
        with open(Path(prog_args.data_input_dir).joinpath(f'{prefx}.{dt}.input')) as fp:
            x_content = fp.read()
            x_content = x_content.splitlines()
        with open(Path(prog_args.data_input_dir).joinpath(f'{prefx}.{dt}.target{postfix}')) as fp:
            y_content = fp.read()
            y_content = y_content.splitlines()

        x = self.tokenizer(
            x_content, padding='max_length', truncation=True,
            max_length=prog_args.model_max_length, return_tensors='pt')
        y = self.tokenizer(
            y_content, padding='max_length', truncation=True,
            max_length=prog_args.model_max_length, return_tensors='pt')

        loader = DataLoader(
            BatchEncodingDataset(x, y),
            batch_size=prog_args.batch_size,
            shuffle=shuffle,
            num_workers=6
        )

        print(f'{dt.title()} input shape:    ', x['input_ids'].shape)
        print(f'{dt.title()} target shape:   ', y['input_ids'].shape)

        return loader

    def train_dataloader(self):
        return self._get_loader('train')

    def val_dataloader(self):
        return self._get_loader('valid')

    def test_dataloader(self):
        return self._get_loader('test')



def create_trainer():
    torch.set_num_threads(prog_args.torch_num_threads)
    lit_model = LitModule()

    ckpt_path = Path(prog_args.ckpt_dir)
    root_dir = Path('.torch-lightning')
    summary_dir = Path(prog_args.summary_dir)

    with SummaryWriter(log_dir=str(summary_dir)) as summary_writer:
        checkpoint_callback = StatefulModelCheckpoint(
            save_top_k=10,
            monitor=prog_args.early_stop_monitor,
            mode=prog_args.early_stop_mode,
            dirpath=ckpt_path,
            filename='{epoch:04d}-{v.loss:.2f}-{v.acc:.2f}-{v.full:.2f}',
            save_on_train_epoch_end=True,
            verbose=True
        )
        early_stopping_callback = EarlyStopping(
            monitor=prog_args.early_stop_monitor,
            min_delta=prog_args.early_stop_min_delta,
            mode=prog_args.early_stop_mode,
            patience=prog_args.early_stop_patience,
            verbose=True
        )
        auto_epoch_end_callback = AutoEpochEndCallbackForLossAccFullAcc(summary_writer)
        auto_batch_end_callback = AutoBatchEndForLM()
        best_model_checkpoint_callback = BestModelCheckpoint(ckpt=checkpoint_callback)
        checkpoint_callback.restore(verbose=True)

        trainer = Trainer(
            max_epochs=prog_args.max_epochs,
            devices=prog_args.devices,
            limit_train_batches=prog_args.limit_train_batches,
            limit_val_batches=prog_args.limit_val_batches,
            limit_test_batches=prog_args.limit_test_batches,
            limit_predict_batches=prog_args.limit_pred_batches,
            accelerator=prog_args.accelerator,
            callbacks=[
                checkpoint_callback,
                early_stopping_callback,
                auto_epoch_end_callback,
                auto_batch_end_callback,
                best_model_checkpoint_callback
            ],
            default_root_dir=str(root_dir)
        )

        if hasattr(checkpoint_callback, 'best_epoch') and checkpoint_callback.best_epoch is not None:
            set_trainer_epoch(trainer, checkpoint_callback.best_epoch + 1)

        lit_model = load_model_or_checkpoint(lit_model=lit_model, checkpoint=checkpoint_callback)

        return trainer, lit_model


def run_trainer():

    trainer, lit_model = create_trainer()
    trainer.fit(model=lit_model)
    trainer.test(model=lit_model)


def save_model():
    trainer, lit_model = create_trainer()
    lit_model.model.save_pretrained(prog_args.model_dir)        


def main():
    run_trainer()
    if not prog_args.no_model_save:
        save_model()
    return 0



if __name__ == '__main__':

    accuracy_queue = deque(maxlen=50)
    precision_queue = deque(maxlen=50)
    recall_queue = deque(maxlen=50)

    # Custom directory for storing log files
    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # The log file name is set according to the program running time
    log_print = open('Logs/Defalust.log', 'w')
    sys.stdout = log_print
    sys.stderr = log_print


    if len(sys.argv) > 1:
        prog_args = arg_parser.parse_args(namespace=Args())
        main()
    else:
        arg_parser.print_help()

# python RADI-PR.py -i cr/ -X t5 -r text -E 1 -tcD model_t5/t5-base/ -mcD model_t5/ --devices 0
