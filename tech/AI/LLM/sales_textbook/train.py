"""
Train a model
"""
import os
import sys
import math
import random
import requests
import pickle
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import tiktoken
import numpy as np
import pandas as pd
# from aim import Run
from model import Model

# Hyperparameters

# How many batches per training step
batch_size = 4
# Length of the token chunk each batch
context_length = 16
# The vector size of the token embeddings
d_model = 64
# Number of heads in Multi-head attention
num_heads = 4
# 我们的代码中通过 d_model / num_heads = 来获取 head_size

# Number of transformer blocks
num_layers = 8

# 0.001
learning_rate = 1e-3
# Dropout rate
dropout = 0.1
# Total of training iterations
max_iters = 5000
# How often to evaluate the model
eval_interval = 50
# How many iterations to average the loss over when evaluating the model
eval_iters = 20

# Instead of using the cpu, we'll use the GPU if it's available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Torch manual randomize seed
TORCH_SEED = 1337
# Generate a random number as a seed
TorchRandSeed = random.randint(-9223372036854775808, 18446744073709551615)
# Or fixed use of a randomize seed
TorchRandSeed = TORCH_SEED
torch.manual_seed(TorchRandSeed)
torch.cuda.manual_seed_all(TorchRandSeed)

# Step 1: Read the dataset
dataset_read_done = False
if os.path.exists('sales_textbook.txt'):
    with open ('sales_textbook.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        if len(text) > 0:
            dataset_read_done = True

# Download the dataset from url, if the file is not exists.
if (not os.path.exists('sales_textbook.txt')) or (not dataset_read_done):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'

    print('Start to downloading url = %s' % url)

    with open('sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

    print('Dataset downloaded done and save to [%s].' % 'sales_textbook.txt')

# Read the dataset
if not dataset_read_done:
    with open ('sales_textbook.txt', 'r', encoding='utf-8') as f:
        text = f.read()

print('Dataset have read done from [%s]' % 'sales_textbook.txt')
print('')

# Step 2: Tokenization

# Using TikToken to tokenize the source text
encoding = tiktoken.get_encoding('cl100k_base')
# size of tokenized source text is 77,919
tokenized_text = encoding.encode(text)
# size of vocabulary is 3,771
vocab_size = len(set(tokenized_text))
max_token_value = max(tokenized_text)
total_tokens = encoding.encode_ordinary(text)

print(f"Tokenized text size: {len(tokenized_text)}")
print(f"Vocabulary size: {vocab_size}")
print(f"The maximum value in the tokenized text is: {max_token_value}")
print(f"数据集合计有 {len(total_tokens):,} tokens")
print('')

tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)

# Split train and validation
split_idx = int(len(tokenized_text) * 0.8)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]

# Prepare data for training batch
data = train_data
idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
x_batch = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
y_batch = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
print(x_batch.shape, x_batch.shape)
print('')

# pd.DataFrame(x_batch[0].numpy())
for i, x in enumerate(x_batch):
    if i < 4:
        print(f'x_batch[{i}] = {encoding.decode(x_batch[i].numpy())}')
    else:
        break

print('')

# Step 3: Word Embedding
input_embedding_lookup_table = nn.Embedding(max_token_value + 1, d_model)
# tensor[4, 16, 64]: [batch_size, context_length, d_model]
x_batch_embedding = input_embedding_lookup_table(x_batch.data)
y_batch_embedding = input_embedding_lookup_table(y_batch.data)

print(input_embedding_lookup_table.weight.data)
print(x_batch_embedding.shape, y_batch_embedding.shape)

# Step 4: Position Embedding
position_encoding_lookup_table = torch.zeros(context_length, d_model)
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
# apply the sine & cosine
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- math.log(10000.0) / d_model))
# PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))
position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
# add batch to the first dimension
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size, -1, -1)

print("Position Encoding Look-up Table: ", position_encoding_lookup_table.shape)
print(position_encoding_lookup_table)
print('')

# Initialize the model
# model = Model().to(device)
