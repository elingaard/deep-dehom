# Deep Dehom

![alt text](https://github.com/elingaard/deep-dehom/blob/main/double_clamped_200_50_vol_0.25_MinMu_0.10_p10.png)

**!!! DISCLAIMER !!!**
This code is experimental and mainly serves as a reference for replicating the results in the paper "De-homogenization using Convolutional Neural Networks" (https://arxiv.org/abs/2105.04232)

## Installation
To train your own model or run one of the pre-trained models start by installing the required packages:
```
conda create -n deepdehom python=3.8 (Optional)
conda activate deepdehom (Optional)
pip3 install -r requirements.txt
```

## Generate the dataset
Since the model is trained on low-resolution synthethic data it's easy to generate the dataset needed to train the model yourself:
```
python3 data_sampler.py --savepath "path/to/data/train" --n_samples 10000
python3 data_sampler.py --savepath "path/to/data/test" --n_samples 1000
```

## Train your own model
As described in the paper the training is performed in two stages. In the first step the forking loss is disabled, while in the second step the frequency loss is disabled. For more information on why this is needed please refer to the paper. Using the weight factors from the paper the two stage training can be run as:

```
step 1: python3 main.py "path/to/data" --lambda_orient 1.0 --lambda_freq 1.0 --lambda_tv 1.0 --lambda_fork 0.0
step 2: python3 main.py "path/to/data" --pretrained "path/to/step1_model/" --lambda_orient 1.0 --lambda_freq 0.0 --lambda_tv 1.0 --lambda_fork 2.0
```

## Run pre-trained model
To try out the pre-trained model a jupyter notebook `lam_width_projection.ipynb` has been provided. Here the pre-trained model can be used with a homogenization design of your own as input, or one of the pre-generated designs located in the `Output_TO` folder.

Pre-trained models and the dataset used to train them is available at: https://data.dtu.dk/collections/Deep_De-Homogenization/5684665/1

## TO-DO
- [x] Use `Fire` for running data-sampler and trainer
- [x] Add type-hints to functions
- [x] Add links to training data and pretrained weights 
- [x] Re-structure the code with `pytorch-lightning`



