'''
Gets the subset of the vocabulary that is most common in the validation set.
'''

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from transformers import AutoTokenizer
import pickle

top_k = 5000

dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

seen_tokens = {}

for row in tqdm(dataset, total=21990):
    text = row['text']

    for t in tokenizer.encode(text):
        if t not in seen_tokens:
            seen_tokens[t] = 0
        seen_tokens[t] += 1

sorted_tokens = sorted(seen_tokens.items(), key=lambda x: x[1], reverse=True)
vocab_subset = [t for t, _ in sorted_tokens[:top_k]]

for t in reversed(vocab_subset):
    print(tokenizer.decode([t]), seen_tokens[t])

label_map = {token: i for i, token in enumerate(vocab_subset)}

with open("vocab_val.pkl", "wb") as f:
    pickle.dump(label_map, f)
