# deep-dehom

![alt text](https://github.com/elingaard/deep-dehom/blob/main/double_clamped_200_50_vol_0.25_MinMu_0.10_p10.png)

**!!! Code will be uploaded soon !!!**

This is the official repository for the paper "De-homogenization using Convolutional Neural Networks" (https://arxiv.org/abs/2105.04232)

## Installation
To train your own model or run one of the pre-trained model start by installing the required packages:
```
conda create -n deepdehom python=3.8 (Optional)
conda activate deepdehom (Optional)
pip3 install -r requirements.txt
```

## Generate the dataset
Since the model is trained on low-resolution synthethic data it's easy to generate the dataset need to train the model yourself:
```
python3 data_sampler.py --savepath "/Users/martinelingaard/repos/deep-dehom/training_data/test" --n_samples 100
```

## Train your own model
As described in the paper the training is performed in two stages. In the first step the forking loss is disabled, while in the second step the frequency loss is disabled. For more information on why this is needed please refer to the paper. Using the weight factors from the paper the two stage training can be run as:

```
step 1: python3 main.py "path/to/data" lambda_orient 1.0 --lambda_freq 1.0 --lambda_tv 1.0 --lambda_fork 0.0
step 2: python3 main.py "path/to/data" --pretrained "path/to/step1_model/" lambda_orient 1.0 --lambda_freq 0.0 --lambda_tv 1.0 --lambda_fork 2.0
```

## Run pre-trained model
To try out the pre-trained a jupyter notebook `lam_width_projection.py` has been provided. Here the pre-trained model can be used with a homogenization design of your own as input, or one of the pre-generated designs located in the `Output_TO` folder.

## TO-DO
- [ ] Use `Fire` for running data-sampler and trainer
- [ ] Add type-hints for all functions
- [ ] Re-structure the code with `pytorch-lightning`



