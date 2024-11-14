from typing import Tuple, Union, List

import numpy as np
import torch
from torch.utils.data import Dataset


# 原始 batchEncoding
class BatchEncodingDataset(Dataset):
    def __init__(self, inputs, targets):
        assert len(inputs['input_ids']) == len(targets['input_ids'])
        assert len(inputs['attention_mask']) == len(targets['attention_mask'])
        self.inp_data = inputs['input_ids']
        self.tar_data = targets['input_ids']
        self.inp_data_mask = inputs['attention_mask']
        self.tar_data_mask = targets['attention_mask']

    def __getitem__(self, index):
        return (
            self.inp_data[index], self.tar_data[index],
            self.inp_data_mask[index], self.tar_data_mask[index]
        )

    def __len__(self):
        return len(self.inp_data)


# 修改后 batchEncoding
# class BatchEncodingDataset(Dataset):
#     def __init__(self, inputs, targets):
#         assert len(inputs['input_ids']) == len(targets['input_ids'])
#         assert len(inputs['attention_mask']) == len(targets['attention_mask'])
#         self.inp_data = inputs['input_ids']
#         self.tar_data = targets['input_ids']
#         self.inp_data_mask = inputs['attention_mask']
#         self.tar_data_mask = targets['attention_mask']

#     def __getitem__(self, index):

#         self.inp_data[index], self.inp_data_mask[index] = self.augment_tokens(self.inp_data[index], self.inp_data_mask[index])

#         return (
#             self.inp_data[index], self.tar_data[index],
#             self.inp_data_mask[index], self.tar_data_mask[index]
#         )

#     def augment_tokens(self, tokens, mask):
#         # 随机选择一种数据增强策略
#         strategies = [self.token_deletion, self.token_insertion, self.token_swap]
#         strategy = random.choice(strategies)
#         return strategy(tokens, mask)

#     # 更新token增强方法以处理遮罩
#     def token_deletion(self, tokens, mask):
#         if len(tokens) > 1:
#             idx = random.randint(0, len(tokens) - 1)
#             tokens = torch.cat((tokens[:idx], tokens[idx+1:]), dim=0)
#             mask = torch.cat((mask[:idx], mask[idx+1:]), dim=0)
#         return tokens, mask

#     def token_insertion(self, tokens, mask):
#         random_token = torch.randint(0, torch.max(tokens) + 1, (1,))
#         idx = random.randint(0, len(tokens))

#         tokens = torch.cat((tokens[:idx], random_token, tokens[idx:]), dim=0)
#         mask = torch.cat((mask[:idx], torch.tensor([1]), mask[idx:]), dim=0)

#         return tokens, mask

#     def token_swap(self, tokens, mask):
#         if len(tokens) > 1:
#             idx1, idx2 = random.sample(range(len(tokens)), 2)
#             tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
#             mask[idx1], mask[idx2] = mask[idx2], mask[idx1]
#         return tokens, mask

#     def __len__(self):
#         return len(self.inp_data)


class InputTargetDataset(Dataset):
    def __init__(
            self,
            inputs: Union[
                np.ndarray,
                torch.Tensor,
                List[Union[np.ndarray, torch.Tensor]],
                Tuple[Union[np.ndarray, torch.Tensor], ...]
            ],
            targets: Union[
                np.ndarray,
                torch.Tensor,
                List[Union[np.ndarray, torch.Tensor]],
                Tuple[Union[np.ndarray, torch.Tensor], ...]
            ]):
        self._is_src_single = self._is_tgt_single = False
        if isinstance(inputs, (np.ndarray, torch.Tensor)):
            self._is_src_single = True
            inputs = (inputs,)
        if isinstance(targets, (np.ndarray, torch.Tensor)):
            self._is_tgt_single = True
            targets = (targets,)

        assert all(len(src) == len(tgt) for src in inputs for tgt in targets), 'Size mismatch between tensors.'

        self.src = inputs
        self.tgt = targets

    def __getitem__(self, item):
        if self._is_src_single:
            inputs = self.src[0][item]
        else:
            inputs = tuple(src[item] for src in self.src)
        if self._is_tgt_single:
            targets = self.tgt[0][item]
        else:
            targets = tuple(tgt[item] for tgt in self.tgt)
        return inputs, targets

    def __len__(self):
        return len(self.src[0]) if len(self.src) > 0 else 0
