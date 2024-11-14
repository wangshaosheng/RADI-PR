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
    '-X', '--experiment', type=str, choices=['t5', 'codet5', 'graph', 'llama'], required=True,
    help='Experiment configuration.'
         'Option t5 defines the experiments on t5-base pretrained model, '
         'while codet5 will use Salesforce/codet5-base. '
         'Graph is based on a composite model described in the paper.')
arg_parser.add_argument(
    '-r', '--representation', type=str, choices=['text', 'cmdseqtoken', 'graphtext'], required=True,
    help='Data representation that will be used during training.')
arg_parser.add_argument(
    '-T', '--task', type=str, choices=['ILPR', 'CLPR'], required=True,
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



import torch
import torch.nn as nn
import torch.nn.functional as F

class RankingCrossEntropyLoss(nn.Module):
    def __init__(self, margin=0.001, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', max_margin_loss=1.0):
        super().__init__()
        self.margin = margin
        self.ignore_index = ignore_index
        self.max_margin_loss = max_margin_loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average,
                                                      ignore_index=ignore_index, reduce=reduce, reduction=reduction)

    def forward(self, logits, targets):
        # 交叉熵损失
        ce_loss = self.cross_entropy_loss(logits.transpose(1, 2).reshape(-1, logits.size(1)), targets.view(-1))

        # 创建mask
        mask = (targets != self.ignore_index).unsqueeze(1).expand(-1, logits.size(1), -1)
        masked_logits = logits * mask.float()
        
        # 防止除以零
        epsilon = 1e-8
        masked_logits = masked_logits + epsilon
        
        # 选择正确类别的logits
        masked_correct_logits = masked_logits.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 计算排序损失
        mean_logits = masked_logits.sum(dim=1) / (mask.sum(dim=1) + epsilon)
        raw_margin_loss = self.margin - (masked_correct_logits - mean_logits)
        
        # 平滑处理：使用Huber损失代替ReLU
        margin_loss = F.smooth_l1_loss(raw_margin_loss, torch.zeros_like(raw_margin_loss), reduction='mean')

        # 限制损失值
        margin_loss = torch.clamp(margin_loss, 0, self.max_margin_loss)

        # 结合两种损失
        combined_loss = ce_loss + margin_loss

        return combined_loss

class CombinedLoss(nn.Module):
    def __init__(self, margin=2.5, ignore_index=-100, max_margin_loss=10.0):
        super().__init__()
        self.margin = margin
        self.ignore_index = ignore_index
        self.max_margin_loss = max_margin_loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        # 计算交叉熵损失
        ce_loss = self.ce_loss(logits.transpose(1, 2).reshape(-1, logits.size(1)), labels.view(-1))

        # 创建一个mask，忽略ignore_index对应的位置
        mask = (labels != self.ignore_index).unsqueeze(2).expand_as(logits.transpose(1, 2))

        # 计算排列边际损失
        batch_size, vocab_size, seq_len = logits.size()
        logits = logits.transpose(1, 2)  # 将形状调整为 [batch_size, sequence_length, vocab_size]

        # 展平labels以用于gather
        flat_labels = labels.view(-1, 1)

        # 得到每个正确标签的logits，并应用mask
        correct_logits = logits.reshape(-1, vocab_size).gather(1, flat_labels).view(batch_size, seq_len)
        correct_logits = correct_logits.unsqueeze(2) * mask

        # 应用mask到logits
        masked_logits = logits * mask

        # 排列边际损失的计算
        pairwise_margin_loss = F.relu(self.margin + masked_logits - correct_logits).sum(dim=2).mean()
        
        # 限制排列边际损失的值
        pairwise_margin_loss = torch.clamp(pairwise_margin_loss, max=self.max_margin_loss)

        # 组合两个损失
        combined_loss = ce_loss + pairwise_margin_loss

        return combined_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

# class CLPRLoss(nn.Module):
#     def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, lambda_4=1.0, ignore_index=-100, reduction='mean'):
#         super(CLPRLoss, self).__init__()
#         self.lambda_1 = lambda_1
#         self.lambda_2 = lambda_2
#         self.lambda_3 = lambda_3
#         self.lambda_4 = lambda_4
#         self.ignore_index = ignore_index
#         self.reduction = reduction
        
#         # define CrossEntropyLoss
#         self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

#     def source_cross_entropy_loss(self, logits, targets):
#         """
#         Calculate the cross entropy loss of the source language
#         """
#         # 交叉熵损失
#         return self.ce_loss(logits.transpose(1, 2).reshape(-1, logits.size(1)), targets.view(-1))

#     def domain_adaptation_loss(self, domain_logits, domain_labels):
#         """
#         Computing domain adaptation loss
#         """
#         domain_loss = F.binary_cross_entropy_with_logits(domain_logits, domain_labels.float())
#         return domain_loss

#     def target_entropy_loss(self, target_prob):
#         """
#         Calculate the entropy loss of the target language
#         """
#         # 计算目标语言的熵
#         entropy_loss = -(target_prob * target_prob.log()).sum(dim=-1).mean()
#         return entropy_loss

#     def forward(self, logits, targets, domain_logits=None, domain_labels=None, target_prob=None):
#         """
#         Calculate the final weighted loss
#         """
#         # 1. 计算源语言交叉熵损失
#         source_loss = self.source_cross_entropy_loss(logits, targets)

#         # 2. 计算领域适配损失（如果提供了领域分类器输出）
#         domain_loss = 0
#         if domain_logits is not None and domain_labels is not None:
#             domain_loss = self.domain_adaptation_loss(domain_logits, domain_labels)

#         # 3. 计算目标语言的熵损失（如果提供了目标语言的概率分布）
#         target_loss = 0
#         if target_prob is not None:
#             target_loss = self.target_entropy_loss(target_prob)

#         # 4. 计算信心损失（这里可以是 confidence 相关的损失函数，根据需要进行调整）
#         # 我们假设这个损失是与 confidence 直接相关的损失
#         confidence_loss = 0
#         # confidence_loss 的计算方式可以根据需要修改

#         # 计算总损失
#         total_loss = (self.lambda_1 * source_loss + 
#                       self.lambda_2 * domain_loss + 
#                       self.lambda_3 * target_loss + 
#                       self.lambda_4 * confidence_loss)

#         return total_loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class CLPRLoss(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, lambda_4=1.0, ignore_index=-100, reduction='mean'):
        super(CLPRLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # 定义 CrossEntropyLoss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def source_cross_entropy_loss(self, logits, targets):
        """
        计算源语言的交叉熵损失
        """
        return self.ce_loss(logits.transpose(1, 2).reshape(-1, logits.size(1)), targets.view(-1))

    def domain_adaptation_loss(self, logits, targets):
        """
        假设 logits 是源领域的输出，targets 是目标领域的标签
        计算源领域与目标领域之间的适配损失
        """
        return self.ce_loss(logits.transpose(1, 2).reshape(-1, logits.size(1)), targets.view(-1))

    def target_entropy_loss(self, target_prob):
        """
        计算目标语言的熵损失，鼓励模型对目标语言的预测更有信心
        """
        entropy_loss = -(target_prob * target_prob.log()).sum(dim=-1).mean()
        return entropy_loss

    def calculate_confidence(self, logits):
        """
        计算信心：对每个token应用softmax并选择最大概率值
        """
        probabilities = F.softmax(logits, dim=-1)
        confidence, _ = torch.max(probabilities, dim=-1)
        return confidence

    def confidence_loss(self, confidence, threshold=0.7):
        """
        计算信心损失，惩罚低于阈值的信心。
        """
        return F.relu(threshold - confidence).mean()

    def forward(self, logits, targets, target_prob=None):
        """
        计算最终的加权损失
        """
        # 1. 计算源语言交叉熵损失
        source_loss = self.source_cross_entropy_loss(logits, targets)

        # 2. 计算领域适配损失
        domain_loss = 0  # 假设没有领域适配损失

        # 3. 计算目标语言的熵损失（如果提供了目标语言的概率分布）
        target_loss = 0
        if target_prob is not None:
            target_loss = self.target_entropy_loss(target_prob)

        # 4. 计算信心损失
        confidence = self.calculate_confidence(logits)
        confidence_loss_value = self.confidence_loss(confidence)

        # 计算总损失
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
        if prog_args.task == 'ILPR':
            self.criterion = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        else:
            self.criterion = CLPRLoss()  # 你可以替换成其他损失函数

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

        # Load vectorized change information
        with open('vectorized_intervention_strategies.json', 'r') as f:
            self.vectorized_strategies = json.load(f)

    def calculate_confidence(self, logits):
        # Use softmax to calculate the confidence of the output
        probabilities = F.softmax(logits, dim=-1)
        confidence, _ = torch.max(probabilities, dim=-1)
        return confidence

    def retrieve_info(self, src_text):
        # Retrieve change information based on input sequence
        retrieved_info = []
        for key in self.vectorized_strategies:
            if key in src_text:
                retrieved_info.append(self.vectorized_strategies[key])
        if not retrieved_info:
            return None

        # Convert the retrieved information into tensor
        input_ids = torch.tensor([info['input_ids'] for info in retrieved_info])
        attention_masks = torch.tensor([info['attention_mask'] for info in retrieved_info])
        labels = torch.tensor([info['labels'] for info in retrieved_info])

        return input_ids, attention_masks, labels



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



    # def forward_batch(self, batch, step_type: str = None):
    #     if prog_args.experiment == 't5' or prog_args.experiment == 'codet5':
    #         src_data, tgt_data, src_data_mask, _ = batch

    #         # Retrieving change information
    #         retrieved_embeddings = []
    #         for src in src_data:
    #             src_text = self.tokenizer.decode(src)
    #             retrieved_info = self.retrieve_info(src_text)
    #             if retrieved_info:
    #                 input_ids, attention_masks, labels = retrieved_info
    #                 embeddings = self.model.shared(input_ids.to(self.device))
    #                 retrieved_embeddings.append((embeddings, labels))

    #         model_output = self(
    #             input_ids=src_data,
    #             labels=tgt_data,
    #             attention_mask=src_data_mask
    #         )
    #         logits = model_output.logits  

    #         # Calculating model confidence
    #         confidence = self.calculate_confidence(logits)  

    #         if retrieved_embeddings:
    #             for embeddings, labels in retrieved_embeddings:
    #                 # Make sure embeddings and labels have the same shape
    #                 embeddings = embeddings.squeeze(0)
    #                 for i in range(labels.size(1)):
    #                     label = labels[0, i]
    #                     if label != self.tokenizer.pad_token_id:
    #                         if confidence < self.confidence_threshold:
    #                             logits[:, i, :] = embeddings[i, :]


    #         logits = logits.transpose(-1, -2)
    #         loss = self.criterion(logits, tgt_data)
    #         loss_val = self.train_loss_metric(loss)
    #         acc_val = self.train_accuracy(logits, tgt_data)
    #         logits = f.lift_predictions(logits, tgt_data, ignore_index=self.tokenizer.pad_token_id)
    #         full_acc_val = self.train_full_accuracy(logits, tgt_data)


    #         return {
    #             'logits': model_output.logits,
    #             'labels': tgt_data,
    #             'loss': loss,
    #             'mean_loss': loss_val,
    #             'accuracy': acc_val,
    #             'full_accuracy': full_acc_val,
    #             'step_type': step_type
    #         }

    #     else:
    #         raise NotImplementedError()

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
            confidence_value = confidence.mean()  # Or use .max(), .min(), or .item() depending on your logic

            if retrieved_embeddings:
                for embeddings, labels in retrieved_embeddings:
                    # Make sure embeddings and labels have the same shape
                    embeddings = embeddings.squeeze(0)
                    for i in range(labels.size(1)):
                        label = labels[0, i]
                        if label != self.tokenizer.pad_token_id:
                            if confidence_value < self.confidence_threshold:
                                logits[:, i, :] = embeddings[i, :]
                            # Optionally adjust logits based on confidence
                            logits[:, i, :] *= confidence_value  # Scaling by confidence

            logits = logits.transpose(-1, -2)

            # Adjust loss based on confidence
            loss = self.criterion(logits, tgt_data)
            if confidence_value < self.confidence_threshold:
                loss *= 1.5  # Increase the loss for low confidence samples
            else:
                loss *= 0.5  # Decrease the loss for high confidence samples

            loss_val = self.train_loss_metric(loss)
            acc_val = self.train_accuracy(logits, tgt_data)
            logits = f.lift_predictions(logits, tgt_data, ignore_index=self.tokenizer.pad_token_id)
            full_acc_val = self.train_full_accuracy(logits, tgt_data)

            return {
                'logits': model_output.logits,
                'labels': tgt_data,
                'loss': loss,
                'mean_loss': loss_val,
                'accuracy': acc_val,
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

        # # data augmentation
        # preprocessor = DataPreprocessor(tokenizer=self.tokenizer, augment=True, max_length=prog_args.model_max_length)
        # x, y = preprocessor.preprocess_data(x, y)

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

# python RADI-PR.py -i cr3/ -X t5 -r text -E 1 -tcD model_t5/t5-base/ -mcD model_t5/ --devices 5 -T ILPR