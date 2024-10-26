
import os
import torch
import torch.nn as nn
import tiktoken
import numpy as np
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
num_blocks = 8

# Instead of using the cpu, we'll use the GPU if it's available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def filepath_filter(filepath):
    # Windows operation is 'nt' or 'windows'
    path_separator = os.sep
    # Whether is windows?
    if path_separator == '\\':
        new_filepath = filepath.replace('/', path_separator)
        return new_filepath
    else:
        return filepath

# Step 1: Read the dataset
dataset_read_ok = False
if os.path.exists('sales_textbook.txt'):
    with open ('sales_textbook.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        if len(text) > 0:
            dataset_read_ok = True

# Step 2: Tokenization

# Using TikToken to tokenize the source text
cl100k_base = tiktoken.get_encoding('cl100k_base')

#
# From: https://github.com/openai/tiktoken/blob/main/README.md
#
# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
cl100k_im = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name = "cl100k_im",
    pat_str = cl100k_base._pat_str,
    mergeable_ranks = cl100k_base._mergeable_ranks,
    special_tokens = {
        **cl100k_base._special_tokens,
        "<|pad|>": 100263,
        "<|start|>": 100264,
        "<|end|>": 100265,
    }
)

# size of tokenized source text is 77,919
tokenized_text = cl100k_im.encode(text)
# size of vocabulary is 3,771
vocab_size = len(set(tokenized_text))
max_token_value = max(tokenized_text)

print(f"Tokenized text size: {len(tokenized_text)}")
print(f"Vocabulary size: {vocab_size}")
print(f"The maximum value in the tokenized text is: {max_token_value}")
print('')

# Step 3: Word Embedding
token_embedding_lookup_table = nn.Embedding(max_token_value + 1, d_model)

# Initialize the model
model = Model(max_token_value + 1).to(device)
state_dict = torch.load(filepath_filter('./model/model-sales-textbook.pt'))
model.load_state_dict(state_dict)

# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型参数量为：{total_params:,}\n")

def word2tokens(text):
    # tokens = []
    # for word in text:
    #     tokens.append(token_embedding_lookup_table(word).numpy())
    tokens = cl100k_im.encode(text)
    print('tokens = ', tokens)
    tokens = torch.tensor(tokens)
    tokens = tokens[:context_length]
    tokens = tokens.unsqueeze(0)
    tokens = tokens.expand(batch_size, -1)
    print('tokens = ', tokens)
    return tokens

def token2word(tokens):
    return cl100k_im.decode(tokens)

if dataset_read_ok:
    while True:
        # ask = input('请输入你的问题: ')
        # ask = "In today's highly competitive market, where customers have numerous options to choose from, it is"
        # ask = "Furthermore, building rapport allows you to differentiate yourself from competitors. "
        ask = "Furthermore, building rapport allows you"
        ask = "The product is a"
        if (ask == 'bye'):
            break
        tokens = word2tokens(ask)
        answers = model.generate(tokens)
        print('answers = ', answers[0])
        print('ask =', ask)
        print('')
        print('answers = ', token2word(answers[0].numpy()))
        print('')
        break
else:
    print('Token file read failed.\n')
