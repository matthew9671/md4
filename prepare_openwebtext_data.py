# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Prepares the input pipeline for OpenWebText (OWT).

This script tokenizes the OWT dataset and splits it into train and eval sets.
The train and eval sets are saved as ArrayRecord files.
"""

from array_record.python import array_record_module
import datasets
import numpy as np
import tensorflow as tf
import tqdm
import transformers
import time
import os
from datasets import load_from_disk

if not os.path.exists("./openwebtext_cache"):

    source = datasets.load_dataset(
        "Skylion007/openwebtext",
        name="plain_text",
        split="train",
        streaming=False,
        num_proc=os.cpu_count(),
    )
    # Save a smaller, cached version
    source.save_to_disk("./openwebtext_cache")

import datasets
from transformers import GPT2Tokenizer
from multiprocessing import Pool
import os
from tqdm import tqdm

# Constants
_GPT2_TOKENIZER = "gpt2"
NUM_WORKERS = os.cpu_count()  # Number of workers (adjust as needed)

# Load the dataset (streaming mode)


def process_chunk(chunk_range):
    """
    Tokenize a chunk of data using skip and take for streaming datasets.

    Args:
        chunk_range (tuple): Start and end indices for the chunk.

    Returns:
        list of dict: Tokenized data.
    """
    start, end = chunk_range
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(_GPT2_TOKENIZER)
    source = datasets.load_dataset(
        "Skylion007/openwebtext", name="plain_text", split="train"
    )
    # Efficiently access the desired range
    chunk = source.select(range(start, end))
    tokenized_data = [tokenizer(text["text"]) for text in tqdm(chunk)]

    return tokenized_data


def create_chunks(total_size, chunk_size):
    """Create chunk ranges for parallel processing."""
    return [
        (i, min(i + chunk_size, total_size)) for i in range(0, total_size, chunk_size)
    ]


total_size = 8013769  # Adjust to the number of samples in your dataset
CHUNK_SIZE = 8013769 // NUM_WORKERS + 1
chunks = create_chunks(total_size, CHUNK_SIZE)

# Run parallel tokenization
with Pool(NUM_WORKERS) as pool:
    tokenized_results = pool.map(process_chunk, chunks)

# Combine and save results
all_tokenized_data = [item for sublist in tokenized_results for item in sublist]
# save_tokenized_data(all_tokenized_data, "./data_dir/openwebtext_tokenized.txt")
import pickle

output_pickle_file = "./openwebtext_tokenized.pkl"
with open(output_pickle_file, "wb") as f:
    pickle.dump(all_tokenized_data, f)

print(
    f"Tokenization complete. Total tokenized entries: {len(all_tokenized_data)}. Saving intermediate results. "
)

all_tokenized_data_input_ids = [data["input_ids"] for data in all_tokenized_data]


_GPT2_TOKENIZER = "gpt2"
tokenizer = transformers.GPT2Tokenizer.from_pretrained(_GPT2_TOKENIZER)

ArrayRecordWriter = array_record_module.ArrayRecordWriter
ArrayRecordReader = array_record_module.ArrayRecordReader


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


ds_output_file_train = "./data_dir/openwebtext_splits_1024_train"
ds_output_file_eval = "./data_dir/openwebtext_splits_1024_eval"

n_examples = 8013769  # tiny: 2; small: 10_000; full: 8013769
save_every_examples = 100_000
block_size = 1024  # size of the chunk

# data_iter = iter(source)

all_tokens = []
count = 0
count_per_save = 0
eval_chunks = []

writer_train = ArrayRecordWriter(ds_output_file_train, "group_size:1")
writer_eval = ArrayRecordWriter(ds_output_file_eval, "group_size:1")

from tqdm import tqdm

for tokens in tqdm(all_tokenized_data_input_ids):
    # tokens = tokenizer(example["text"])["input_ids"]
    all_tokens.extend(tokens + [tokenizer.eos_token_id])
    count += 1
    count_per_save += 1

    # pause to save when having tokenized enough examples for saving.
    time1 = time.time()
    if count_per_save >= save_every_examples:
        # save to disk
        saved_length = (len(all_tokens) // block_size) * block_size
        chunks = [
            all_tokens[i : i + block_size] for i in range(0, saved_length, block_size)
        ]

        print("Time taken to tokenize:", time.time() - time1)
        print(f"\nsaving chunks @ {count}th example mark...")
        np.random.shuffle(chunks)
        num_eval = int(len(chunks) * 0.02)  # put 2% of chunks into eval split.
        for eval_i in tqdm(range(num_eval)):
            feature = {
                "text": _int64_feature(chunks[eval_i]),
            }
            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer_eval.write(example_proto.SerializeToString())

        for train_i in tqdm(range(num_eval, len(chunks))):
            feature = {
                "text": _int64_feature(chunks[train_i]),
            }
            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer_train.write(example_proto.SerializeToString())
        print("Time taken to save:", time.time() - time1)
        # prepare for the next round of tokenize-n-save.
        all_tokens = all_tokens[saved_length:]
        count_per_save = 0

    # stop when having tokenized enough examples for total #.
    if count >= n_examples:
        # save to disk
        saved_length = (len(all_tokens) // block_size) * block_size
        chunks = [
            all_tokens[i : i + block_size] for i in range(0, saved_length, block_size)
        ]

        print(f"\nsaving chunks @ {count}th example mark...")
        np.random.shuffle(chunks)
        num_eval = int(len(chunks) * 0.02)  # put 2% of chunks into eval split.
        for eval_i in tqdm(range(num_eval)):
            feature = {
                "text": _int64_feature(chunks[eval_i]),
            }
            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer_eval.write(example_proto.SerializeToString())

        for train_i in tqdm(range(num_eval, len(chunks))):
            feature = {
                "text": _int64_feature(chunks[train_i]),
            }
            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer_train.write(example_proto.SerializeToString())
        break

writer_train.close()
writer_eval.close()
