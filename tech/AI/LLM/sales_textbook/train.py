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

def filepath_filter(filepath):
    # Windows operation is 'nt' or 'windows'
    path_separator = os.sep
    # Whether is windows?
    if path_separator == '\\':
        new_filepath = filepath.replace('/', path_separator)
        return new_filepath
    else:
        return filepath

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
num_blocks = 8

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
if torch.cuda.is_available():
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

# 将 77,919 个 tokens 转换到 Pytorch 张量中
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
token_embedding_lookup_table = nn.Embedding(max_token_value + 1, d_model)
# tensor[4, 16, 64]: [batch_size, context_length, d_model]
x_batch_embedding = token_embedding_lookup_table(x_batch.data)
y_batch_embedding = token_embedding_lookup_table(y_batch.data)

# print(token_embedding_lookup_table.weight.data)
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

# Add positional encoding into the input embedding vector
input_embedding_x = x_batch_embedding + position_encoding_lookup_table # [4, 16, 64] [batch_size, context_length, d_model]
input_embedding_y = y_batch_embedding + position_encoding_lookup_table

x = input_embedding_x
y = input_embedding_y

x_plot = input_embedding_x[0].detach().cpu().numpy()
print("Final Input Embedding of x: \n", pd.DataFrame(x_plot))
print('')

# Step 5: Transformer Block

# Initialize the model
model = Model().to(device)

# get batch
def get_batch_data(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y

# calculate the loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch_data(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:', round(losses['valid'].item(), 3))
        # run.track(round(losses['train'].item(), 3), name='Training Loss')
        # run.track(round(losses['valid'].item(), 3), name='Validation Loss')

    xb, yb = get_batch_data('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model
model_dir = filepath_filter('./model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.save(model.state_dict(), filepath_filter('{}/model-sales-textbook.pt'.format(model_dir)))

'''
Output:

Step: 0 Training Loss: 11.752 Validation Loss: 11.716
Step: 20 Training Loss: 10.317 Validation Loss: 10.339
Step: 40 Training Loss: 8.675 Validation Loss: 8.932
Step: 60 Training Loss: 7.212 Validation Loss: 7.655
Step: 80 Training Loss: 6.962 Validation Loss: 7.286
Step: 100 Training Loss: 6.63 Validation Loss: 7.379
Step: 120 Training Loss: 6.519 Validation Loss: 7.095
Step: 140 Training Loss: 6.351 Validation Loss: 7.017
Step: 160 Training Loss: 6.535 Validation Loss: 7.008
Step: 180 Training Loss: 6.406 Validation Loss: 6.596
Step: 200 Training Loss: 6.025 Validation Loss: 6.871
Step: 220 Training Loss: 6.171 Validation Loss: 6.697
Step: 240 Training Loss: 6.03 Validation Loss: 6.597
Step: 260 Training Loss: 6.016 Validation Loss: 6.407
Step: 280 Training Loss: 5.785 Validation Loss: 6.361
Step: 300 Training Loss: 5.593 Validation Loss: 6.534
Step: 320 Training Loss: 5.734 Validation Loss: 6.458
Step: 340 Training Loss: 5.707 Validation Loss: 6.281
Step: 360 Training Loss: 5.496 Validation Loss: 6.222
Step: 380 Training Loss: 5.506 Validation Loss: 6.141
Step: 400 Training Loss: 5.243 Validation Loss: 6.26
Step: 420 Training Loss: 5.577 Validation Loss: 6.033
Step: 440 Training Loss: 5.138 Validation Loss: 6.117
Step: 460 Training Loss: 5.131 Validation Loss: 6.093
Step: 480 Training Loss: 4.927 Validation Loss: 5.797
Step: 500 Training Loss: 5.048 Validation Loss: 5.986
Step: 520 Training Loss: 5.225 Validation Loss: 5.786
Step: 540 Training Loss: 5.116 Validation Loss: 6.162
Step: 560 Training Loss: 5.038 Validation Loss: 5.807
Step: 580 Training Loss: 4.68 Validation Loss: 5.798
Step: 600 Training Loss: 4.888 Validation Loss: 5.835
Step: 620 Training Loss: 4.944 Validation Loss: 5.803
Step: 640 Training Loss: 4.681 Validation Loss: 5.703
Step: 660 Training Loss: 4.83 Validation Loss: 6.149
Step: 680 Training Loss: 4.745 Validation Loss: 6.066
Step: 700 Training Loss: 4.92 Validation Loss: 5.795
Step: 720 Training Loss: 4.893 Validation Loss: 5.755
Step: 740 Training Loss: 4.567 Validation Loss: 5.859
Step: 760 Training Loss: 4.753 Validation Loss: 5.367
Step: 780 Training Loss: 4.528 Validation Loss: 5.621
Step: 800 Training Loss: 4.427 Validation Loss: 5.57
Step: 820 Training Loss: 4.491 Validation Loss: 5.51
Step: 840 Training Loss: 4.506 Validation Loss: 5.408
Step: 860 Training Loss: 4.29 Validation Loss: 5.55
Step: 880 Training Loss: 4.569 Validation Loss: 5.533
Step: 900 Training Loss: 4.499 Validation Loss: 5.542
Step: 920 Training Loss: 4.523 Validation Loss: 5.456
Step: 940 Training Loss: 4.202 Validation Loss: 5.787
Step: 960 Training Loss: 4.329 Validation Loss: 5.442
Step: 980 Training Loss: 4.417 Validation Loss: 5.219
Step: 1000 Training Loss: 4.276 Validation Loss: 5.445
Step: 1020 Training Loss: 4.375 Validation Loss: 5.541
Step: 1040 Training Loss: 4.383 Validation Loss: 5.26
Step: 1060 Training Loss: 4.238 Validation Loss: 5.449
Step: 1080 Training Loss: 4.024 Validation Loss: 5.798
Step: 1100 Training Loss: 4.194 Validation Loss: 5.542
Step: 1120 Training Loss: 4.348 Validation Loss: 5.368
Step: 1140 Training Loss: 4.086 Validation Loss: 5.554
Step: 1160 Training Loss: 4.156 Validation Loss: 5.48
Step: 1180 Training Loss: 4.079 Validation Loss: 5.289
Step: 1200 Training Loss: 4.221 Validation Loss: 5.336
Step: 1220 Training Loss: 3.993 Validation Loss: 5.426
Step: 1240 Training Loss: 4.023 Validation Loss: 5.291
Step: 1260 Training Loss: 4.072 Validation Loss: 5.412
Step: 1280 Training Loss: 4.083 Validation Loss: 5.321
Step: 1300 Training Loss: 4.047 Validation Loss: 5.461
Step: 1320 Training Loss: 4.184 Validation Loss: 5.087
Step: 1340 Training Loss: 4.067 Validation Loss: 5.132
Step: 1360 Training Loss: 3.98 Validation Loss: 5.4
Step: 1380 Training Loss: 4.051 Validation Loss: 5.275
Step: 1400 Training Loss: 3.783 Validation Loss: 4.971
Step: 1420 Training Loss: 3.909 Validation Loss: 5.305
Step: 1440 Training Loss: 3.849 Validation Loss: 5.391
Step: 1460 Training Loss: 3.885 Validation Loss: 5.188
Step: 1480 Training Loss: 4.021 Validation Loss: 5.359
Step: 1500 Training Loss: 4.064 Validation Loss: 4.838
Step: 1520 Training Loss: 4.029 Validation Loss: 5.386
Step: 1540 Training Loss: 3.808 Validation Loss: 5.008
Step: 1560 Training Loss: 3.976 Validation Loss: 5.503
Step: 1580 Training Loss: 4.05 Validation Loss: 5.399
Step: 1600 Training Loss: 3.811 Validation Loss: 5.477
Step: 1620 Training Loss: 3.864 Validation Loss: 5.274
Step: 1640 Training Loss: 3.896 Validation Loss: 5.012
Step: 1660 Training Loss: 3.749 Validation Loss: 5.209
Step: 1680 Training Loss: 3.845 Validation Loss: 5.237
Step: 1700 Training Loss: 3.721 Validation Loss: 4.812
Step: 1720 Training Loss: 3.795 Validation Loss: 4.94
Step: 1740 Training Loss: 3.674 Validation Loss: 5.015
Step: 1760 Training Loss: 3.566 Validation Loss: 5.08
Step: 1780 Training Loss: 3.502 Validation Loss: 5.198
Step: 1800 Training Loss: 3.563 Validation Loss: 5.17
Step: 1820 Training Loss: 3.787 Validation Loss: 4.999
Step: 1840 Training Loss: 3.742 Validation Loss: 5.286
Step: 1860 Training Loss: 3.723 Validation Loss: 4.94
Step: 1880 Training Loss: 3.774 Validation Loss: 5.196
Step: 1900 Training Loss: 3.81 Validation Loss: 4.757
Step: 1920 Training Loss: 3.521 Validation Loss: 5.371
Step: 1940 Training Loss: 3.868 Validation Loss: 5.316
Step: 1960 Training Loss: 3.594 Validation Loss: 5.42
Step: 1980 Training Loss: 3.828 Validation Loss: 5.072
Step: 2000 Training Loss: 3.491 Validation Loss: 5.033
Step: 2020 Training Loss: 3.496 Validation Loss: 5.065
Step: 2040 Training Loss: 3.672 Validation Loss: 5.02
Step: 2060 Training Loss: 3.592 Validation Loss: 5.012
Step: 2080 Training Loss: 3.529 Validation Loss: 5.089
Step: 2100 Training Loss: 3.591 Validation Loss: 4.777
Step: 2120 Training Loss: 3.74 Validation Loss: 5.024
Step: 2140 Training Loss: 3.639 Validation Loss: 4.826
Step: 2160 Training Loss: 3.579 Validation Loss: 5.401
Step: 2180 Training Loss: 3.563 Validation Loss: 4.987
Step: 2200 Training Loss: 3.57 Validation Loss: 5.347
Step: 2220 Training Loss: 3.714 Validation Loss: 4.855
Step: 2240 Training Loss: 3.327 Validation Loss: 4.887
Step: 2260 Training Loss: 3.584 Validation Loss: 5.299
Step: 2280 Training Loss: 3.574 Validation Loss: 4.859
Step: 2300 Training Loss: 3.618 Validation Loss: 5.01
Step: 2320 Training Loss: 3.522 Validation Loss: 5.202
Step: 2340 Training Loss: 3.503 Validation Loss: 4.922
Step: 2360 Training Loss: 3.368 Validation Loss: 5.045
Step: 2380 Training Loss: 3.344 Validation Loss: 4.625
Step: 2400 Training Loss: 3.665 Validation Loss: 4.729
Step: 2420 Training Loss: 3.508 Validation Loss: 4.697
Step: 2440 Training Loss: 3.275 Validation Loss: 4.993
Step: 2460 Training Loss: 3.304 Validation Loss: 4.879
Step: 2480 Training Loss: 3.541 Validation Loss: 5.086
Step: 2500 Training Loss: 3.524 Validation Loss: 4.938
Step: 2520 Training Loss: 3.56 Validation Loss: 5.075
Step: 2540 Training Loss: 3.258 Validation Loss: 5.144
Step: 2560 Training Loss: 3.561 Validation Loss: 5.033
Step: 2580 Training Loss: 3.346 Validation Loss: 4.873
Step: 2600 Training Loss: 3.151 Validation Loss: 4.915
Step: 2620 Training Loss: 3.482 Validation Loss: 5.222
Step: 2640 Training Loss: 3.439 Validation Loss: 4.961
Step: 2660 Training Loss: 3.482 Validation Loss: 4.858
Step: 2680 Training Loss: 3.295 Validation Loss: 4.837
.........

Step: 4880 Training Loss: 2.703 Validation Loss: 4.89
Step: 4900 Training Loss: 2.875 Validation Loss: 4.91
Step: 4920 Training Loss: 2.932 Validation Loss: 4.941
Step: 4940 Training Loss: 2.854 Validation Loss: 5.247
Step: 4960 Training Loss: 2.786 Validation Loss: 4.704
Step: 4980 Training Loss: 2.851 Validation Loss: 4.753
Step: 4999 Training Loss: 2.859 Validation Loss: 4.829

'''
