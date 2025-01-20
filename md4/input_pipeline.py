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

"""Deterministic input pipeline."""

from collections.abc import Sequence
import dataclasses
import os
from typing import Any, Union
import urllib.request
import zipfile

import grain.python as grain
import jax
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


# pylint: disable=g-import-not-at-top
try:
    import cv2
except ImportError:
    print("cv2 not found")
FlatFeatures = dict[str, Any]

_DataSet = Union[grain.MapDataset, grain.DataLoader, grain.IterDataset]

_GPT2_TOKENIZER = "gpt2"
_OWT_DATASETS = dict(
    # OSS version. Please prepare the OWT datasets using the following command:
    # python ./prepare_openwebtext_data.py
    dataset_train_path=("./data_dir/openwebtext_splits_1024_train"),
    dataset_eval_path=("./data_dir/openwebtext_splits_1024_eval"),
)


class ChunkDataSource(grain.RandomAccessDataSource):
    """Chunk text data source."""

    def __init__(self, tensor, chunk_size=256, overlapping=False):
        self.chunk_size = chunk_size
        self.overlapping = overlapping
        tensor = tensor.encode("utf-8")
        if not overlapping:
            extra_len = len(tensor) % chunk_size
            if extra_len > 0:
                tensor = tensor[:-extra_len]
            self.tensor = np.array(list(tensor)).reshape(-1, chunk_size)
        else:
            self.tensor = tensor

    def __len__(self):
        if not self.overlapping:
            return self.tensor.shape[0]
        else:
            return len(self.tensor) - self.chunk_size + 1

    def __getitem__(self, record_key):
        if not self.overlapping:
            return {"text": self.tensor[record_key]}
        else:
            start_idx = record_key
            end_idx = record_key + self.chunk_size
            chunk = self.tensor[start_idx:end_idx]
            return {"text": chunk}

    def __repr__(self) -> str:
        return f"ChunkDataSource(len={len(self)},overlapping={self.overlapping})"


def get_data_shape(config):
    return config.data_shape


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
    """Parse serialized example."""

    def __init__(self, data_column):
        self.data_column = data_column

    def map(self, features):
        def _parse(example):
            with tf.device("/CPU:0"):
                parsed = tf.io.parse_example(
                    example,
                    {
                        self.data_column: tf.io.FixedLenSequenceFeature(
                            [], dtype=tf.int64, allow_missing=True
                        )
                    },
                )
                return parsed

        return _parse(features)


def get_num_train_steps(config: config_dict.ConfigDict) -> int:
    """Calculates the total number of training steps."""
    if config.num_train_steps > 0:
        return config.num_train_steps
    # From the beginning. We first shard the data (shard by process_count), then
    # combine all epochs, batch for all local devices.
    # In all steps we would drop the remainder (hence the use of integer
    # division).
    # When start_index is 0 the train_ds.cardinality() and num_train_steps should
    # be equivalent.
    if config.task_type == "image":
        tfds_info = tfds.builder(config.dataset).info
        num_train_records = tfds_info.splits["train"].num_examples
        return int(num_train_records // jax.process_count() * config.num_epochs) // (
            config.per_device_batch_size * jax.local_device_count()
        )
    else:
        raise NotImplementedError()


def create_datasets(
    config: config_dict.ConfigDict, seed: int
) -> tuple[_DataSet, dict[str, _DataSet], dict[str, Any]]:
    """Create Grain data loaders for training and evaluation.

    Args:
      config: Configuration to use.
      seed: Seed for shuffle and random operations in the training dataset.

    Returns:
      A tuple with the training dataset loader, the evaluation dataset
      loader, and a dictionary of other infos.
    """
    info = {}
    assert config.batch_size % jax.process_count() == 0
    process_batch_size = config.batch_size // jax.process_count()
    eval_batch_size = config.get("eval_batch_size", config.batch_size)
    process_eval_batch_size = eval_batch_size // jax.process_count()

    if config.dataset == "openwebtext":
        # we need to pretrain a GPT2 size model with context length of 1024.
        seq_len = config.data_shape[0]
        assert seq_len == 1024
        train_transformations = [ParseFeatures(data_column="text")]
        eval_transformations = [ParseFeatures(data_column="text")]

        train_table_path = _OWT_DATASETS["dataset_train_path"]
        train_source = grain.ArrayRecordDataSource(paths=train_table_path)

        eval_source = {
            "owt_eval": grain.ArrayRecordDataSource(
                paths=_OWT_DATASETS["dataset_eval_path"]
            ),
        }

        tokenizer = transformers.GPT2Tokenizer.from_pretrained(_GPT2_TOKENIZER)
        info["tokenizer"] = tokenizer
    else:
        raise NotImplementedError("Unsupported datasets.")

    train_loader = grain.load(
        source=train_source,
        shuffle=True,
        seed=seed,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        transformations=train_transformations,
        batch_size=process_batch_size,
        worker_count=config.grain_num_workers,
    )

    if config.eval_pad_last_batch:
        raise NotImplementedError(
            "BatchWithPadElements is not implemented in PyGrain yet."
        )
    else:
        drop_remainder = True
        shard_options = grain.ShardByJaxProcess(drop_remainder=drop_remainder)

    eval_loaders = {}
    for split in eval_source:
        eval_loader = grain.load(
            source=eval_source[split],
            num_epochs=1,
            shard_options=shard_options,
            transformations=eval_transformations,
            batch_size=process_eval_batch_size,
            worker_count=0,
            drop_remainder=drop_remainder,
        )
        eval_loaders[split] = eval_loader

    return train_loader, eval_loaders, info
