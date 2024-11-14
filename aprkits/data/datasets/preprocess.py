import random
import torch


# 原始数据增强
# class DataPreprocessor:
#     def __init__(self, augment=True, max_length=None):
#         self.augment = augment
#         self.max_length = max_length

#     def preprocess_data(self, inputs, targets):
#         augmented_inputs = []
#         augmented_targets = []

#         for input_ids, target_ids in zip(inputs['input_ids'], targets['input_ids']):
#             if self.augment:
#                 input_ids = self.augment_tokens(input_ids)

#             # 裁剪或填充序列
#             input_ids = self.pad_or_trim(input_ids, self.max_length)
#             target_ids = self.pad_or_trim(target_ids, self.max_length)

#             augmented_inputs.append(input_ids)
#             augmented_targets.append(target_ids)

#         return {
#             'input_ids': torch.stack(augmented_inputs),
#             'attention_mask': self.generate_attention_masks(augmented_inputs)
#         }, {
#             'input_ids': torch.stack(augmented_targets),
#             'attention_mask': self.generate_attention_masks(augmented_targets)
#         }

#     def pad_or_trim(self, ids, max_length):
#         if len(ids) > max_length:
#             return ids[:max_length]
#         elif len(ids) < max_length:
#             padding = [0] * (max_length - len(ids))  # 假设0是padding的ID
#             return torch.cat((ids, torch.tensor(padding, dtype=ids.dtype)))
#         return ids

#     def generate_attention_masks(self, sequences):
#         masks = [[1 if token_id != 0 else 0 for token_id in seq] for seq in sequences]
#         return torch.tensor(masks)

#     def augment_tokens(self, tokens):
#         strategies = [self.token_deletion, self.token_insertion, self.token_swap]
#         strategy = random.choice(strategies)
#         return strategy(tokens)

#     def token_deletion(self, tokens):
#         if len(tokens) > 1:
#             idx = random.randint(0, len(tokens) - 1)
#             tokens = torch.cat((tokens[:idx], tokens[idx+1:]))
#         return tokens

#     def token_insertion(self, tokens):
#         random_token = torch.randint(0, torch.max(tokens) + 1, (1,))
#         idx = random.randint(0, len(tokens))
#         tokens = torch.cat((tokens[:idx], random_token, tokens[idx:]))
#         return tokens

#     def token_swap(self, tokens):
#         if len(tokens) > 1:
#             idx1, idx2 = random.sample(range(len(tokens)), 2)
#             tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
#         return tokens


# # 优化后的数据增强
# DALO-APR
# import random
# import torch

# class DataPreprocessor:
#     def __init__(self, augment=True, max_length=None):
#         self.augment = augment
#         self.max_length = max_length

#     def preprocess_data(self, inputs, targets):
#         processed_inputs = []
#         processed_targets = []

#         for input_ids, target_ids in zip(inputs['input_ids'], targets['input_ids']):
#             # 处理并添加原始序列
#             original_input_ids = self.pad_or_trim(input_ids, self.max_length)
#             original_target_ids = self.pad_or_trim(target_ids, self.max_length)
#             processed_inputs.append(original_input_ids)
#             processed_targets.append(original_target_ids)

#             if self.augment:
#                 # 处理并添加增强序列
#                 augmented_input_ids = self.augment_tokens(input_ids)
#                 augmented_input_ids = self.pad_or_trim(augmented_input_ids, self.max_length)
#                 # 假设目标序列不需要增强
#                 augmented_target_ids = self.pad_or_trim(target_ids, self.max_length)
#                 processed_inputs.append(augmented_input_ids)
#                 processed_targets.append(augmented_target_ids)

#         return {
#             'input_ids': torch.stack(processed_inputs),
#             'attention_mask': self.generate_attention_masks(processed_inputs)
#         }, {
#             'input_ids': torch.stack(processed_targets),
#             'attention_mask': self.generate_attention_masks(processed_targets)
#         }

#     def pad_or_trim(self, ids, max_length):
#         if len(ids) > max_length:
#             return ids[:max_length]
#         elif len(ids) < max_length:
#             padding = [0] * (max_length - len(ids))  # 假设0是padding的ID
#             return torch.cat((ids, torch.tensor(padding, dtype=ids.dtype)))
#         return ids

#     def generate_attention_masks(self, sequences):
#         masks = [[1 if token_id != 0 else 0 for token_id in seq] for seq in sequences]
#         return torch.tensor(masks)

#     def augment_tokens(self, tokens):
#         strategies = [self.token_deletion, self.token_insertion, self.token_swap]
#         strategy = random.choice(strategies)
#         return strategy(tokens)

#     def token_deletion(self, tokens):
#         if len(tokens) > 1:
#             idx = random.randint(0, len(tokens) - 1)
#             tokens = torch.cat((tokens[:idx], tokens[idx+1:]))
#         return tokens

#     def token_insertion(self, tokens):
#         random_token = torch.randint(0, torch.max(tokens) + 1, (1,))
#         idx = random.randint(0, len(tokens))
#         tokens = torch.cat((tokens[:idx], random_token, tokens[idx:]))
#         return tokens

#     def token_swap(self, tokens):
#         if len(tokens) > 1:
#             idx1, idx2 = random.sample(range(len(tokens)), 2)
#             tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
#         return tokens

class DataPreprocessor:
    def __init__(self, tokenizer, augment=True, max_length=None):
        self.augment = augment
        self.max_length = max_length
        self.tokenizer = tokenizer

    def preprocess_data(self, inputs, targets):
        processed_inputs = []
        processed_targets = []

        for input_ids, target_ids in zip(inputs['input_ids'], targets['input_ids']):
            # 处理并添加原始序列
            original_input_ids = self.pad_or_trim(input_ids, self.max_length)
            original_target_ids = self.pad_or_trim(target_ids, self.max_length)
            processed_inputs.append(original_input_ids)
            processed_targets.append(original_target_ids)

            if self.augment:
                # 处理并添加增强序列
                augmented_input_ids = self.augment_tokens(input_ids)
                augmented_input_ids = self.pad_or_trim(augmented_input_ids, self.max_length)
                # 假设目标序列不需要增强
                augmented_target_ids = self.pad_or_trim(target_ids, self.max_length)
                processed_inputs.append(augmented_input_ids)
                processed_targets.append(augmented_target_ids)

        return {
            'input_ids': torch.stack(processed_inputs),
            'attention_mask': self.generate_attention_masks(processed_inputs)
        }, {
            'input_ids': torch.stack(processed_targets),
            'attention_mask': self.generate_attention_masks(processed_targets)
        }

    def pad_or_trim(self, ids, max_length):
        if len(ids) > max_length:
            return ids[:max_length]
        elif len(ids) < max_length:
            padding = [0] * (max_length - len(ids))  # 假设0是padding的ID
            return torch.cat((ids, torch.tensor(padding, dtype=ids.dtype)))
        return ids

    def generate_attention_masks(self, sequences):
        masks = [[1 if token_id != 0 else 0 for token_id in seq] for seq in sequences]
        return torch.tensor(masks)

    def augment_tokens(self, tokens):
        strategies = [self.token_deletion, self.token_insertion, self.token_swap]
        strategy = random.choice(strategies)
        return strategy(tokens)

    def token_deletion(self, tokens):
        if len(tokens) > 1:
            idx = random.randint(0, len(tokens) - 1)
            tokens = torch.cat((tokens[:idx], tokens[idx+1:]))
        return tokens

    def token_insertion(self, tokens):
        random_token = torch.randint(0, torch.max(tokens) + 1, (1,))
        idx = random.randint(0, len(tokens))
        tokens = torch.cat((tokens[:idx], random_token, tokens[idx:]))
        return tokens

    def token_swap(self, tokens):
        if len(tokens) > 1:
            idx1, idx2 = random.sample(range(len(tokens)), 2)
            tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
        return tokens

    def get_bug_fix_indices(self, tokens):
        bug_start_token_id = self.tokenizer.convert_tokens_to_ids('<BUGS>')
        bug_end_token_id = self.tokenizer.convert_tokens_to_ids('<BUGE>')
        fix_start_token_id = self.tokenizer.convert_tokens_to_ids('<FIXS>')
        fix_end_token_id = self.tokenizer.convert_tokens_to_ids('<FIXE>')

        bug_start_index = (tokens == bug_start_token_id).nonzero(as_tuple=True)
        bug_end_index = (tokens == bug_end_token_id).nonzero(as_tuple=True)
        fix_start_index = (tokens == fix_start_token_id).nonzero(as_tuple=True)
        fix_end_index = (tokens == fix_end_token_id).nonzero(as_tuple=True)

        return bug_start_index, bug_end_index, fix_start_index, fix_end_index





# class DataPreprocessor:
#     def __init__(self, augment=True, max_length=None):
#         self.max_length = max_length

#     def preprocess_data(self, inputs, targets):
#         processed_inputs = []
#         processed_targets = []

#         for input_ids, target_ids in zip(inputs['input_ids'], targets['input_ids']):
#             # 处理并添加原始序列
#             original_input_ids = self.pad_or_trim(input_ids, self.max_length)
#             original_target_ids = self.pad_or_trim(target_ids, self.max_length)
#             processed_inputs.append(original_input_ids)
#             processed_targets.append(original_target_ids)

#         return {
#             'input_ids': torch.stack(processed_inputs),
#             'attention_mask': self.generate_attention_masks(processed_inputs)
#         }, {
#             'input_ids': torch.stack(processed_targets),
#             'attention_mask': self.generate_attention_masks(processed_targets)
#         }

#     def pad_or_trim(self, ids, max_length):
#         if len(ids) > max_length:
#             return ids[:max_length]
#         elif len(ids) < max_length:
#             padding = [0] * (max_length - len(ids))  # 假设0是padding的ID
#             return torch.cat((ids, torch.tensor(padding, dtype=ids.dtype)))
#         return ids

#     def generate_attention_masks(self, sequences):
#         masks = [[1 if token_id != 0 else 0 for token_id in seq] for seq in sequences]
#         return torch.tensor(masks)