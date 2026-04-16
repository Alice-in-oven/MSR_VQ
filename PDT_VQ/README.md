# PDT-VQ: Boosting Deep Vector Quantization with Progressive Distribution Transformation

This repository contains the offitial implementation of the KDD'25 paper "Boosting Deep Vector Quantization with Progressive Distribution Transformation".

## Install
To set up the environment for PDT, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Lenovo-ICI/PDT-VQ
cd PDT-VQ
# Create the conda environment using the provided configuration
conda env create -f environment.yml
```

## Data preparation

Download all datasets (BigANN, Deep, Sift1M, Gist1M) with:

```
sh download_data.sh
```

Or specify one or more individual datasets (sift1m, gist1m, deep, bigann): 

```
sh download_data.sh <dataset_name_1> <dataset_name_2> ...
```

## Training and Evaluation

Run training with:

```
sh scripts/exp_<dataset_name>_<vq_type>.sh
```

Run evaluation only with:

```
python val.py --exp_path <exp_path>
```

where `<exp_path>` is the path of experiment logs.


## References

```
@inproceedings{pdt,
    title={Boosting Deep Vector Quantization with Progressive Distribution Transformation},
    author={Weikang Wang, Xin Zhou, Jun Liu, Weifeng Zhang, and Huan Yan},
    booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    year={2025}
}
```