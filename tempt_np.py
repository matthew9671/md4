import pickle
import numpy as np

output_pickle_file = "./openwebtext_tokenized_inputids.pkl"
with open(output_pickle_file, "rb") as f:
    all_tokenized_data_input_ids = pickle.load(f)

# ArrayRecordWriter = array_record_module.ArrayRecordWriter
# ArrayRecordReader = array_record_module.ArrayRecordReader


# def _int64_feature(value):
#     """Returns an int64_list from a bool / enum / int / uint."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
import transformers
import numpy as np

_GPT2_TOKENIZER = "gpt2"
tokenizer = transformers.GPT2Tokenizer.from_pretrained(_GPT2_TOKENIZER)

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

# writer_train = ArrayRecordWriter(ds_output_file_train, "group_size:1")
# writer_eval = ArrayRecordWriter(ds_output_file_eval, "group_size:1")

from tqdm import tqdm

for tokens in tqdm(all_tokenized_data_input_ids):
    # tokens = tokenizer(example["text"])["input_ids"]
    all_tokens.extend(tokens + [tokenizer.eos_token_id])
    count += 1
    count_per_save += 1

all_tokens = np.array(all_tokens)
np.save("openwebtext_tokenized_inputids.npy", all_tokens)
