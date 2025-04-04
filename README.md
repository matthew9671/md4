# MD4: Simplified and Generalized Masked Diffusion for Discrete Data


## Installation

### Create Python environment

please use `requirements_gpu.txt` if your accelerator is GPUs, use
`requirements_tpu`.txt when using Google Cloud TPUs.

```
conda create -y --name md4_venv python=3.10
conda activate md4_venv
pip install -r requirements_gpu.txt
export PYTHONPATH="$PYTHONPATH:~/md4"
```

```
cd md4
mkdir data_dir
cd data_dir
gsutil cp -r gs://maskdiff/protein/ .

```

## Usage

prepare openwebtext for training (i.e., tokenize and pack examples)

```
mkdir data_dir
python prepare_openwebtext_data.py
```

train a MD4-S model over text data (using openwebtext).

```
python md4/main.py --config=md4/configs/md4/openwebtext.py --sharded=false --workdir=./expt
```

alternatively, you can train a MD4-S model over image data (using cifar10).

```
python md4/main.py --config=md4/configs/md4/cifar10.py --sharded=false --workdir=./expt
```

### choose batch size

Batch size depends on your compute resource. For training a MD4-S model with
sequence length 1024, eight `A100` GPUs can support a maximum batch size of
`128`. If running on TPUs, eight `v5litepod` chips can support a maximum batch
size of `32`.

## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

```latex
@inproceedings{shi2024simplified,
  title={Simplified and Generalized Masked Diffusion for Discrete Data},
  author={Shi, Jiaxin and Han, Kehang and Wang, Zhe and Doucet, Arnaud and Titsias, Michalis K.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
