# Cassava Leaf Disease Classification

Top-1% solution to the [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) Kaggle competition on plant image classification.

![sample](https://i.postimg.cc/jdtWjXyF/cassava-sample.png)


## Summary

Cassava is one of the key food crops grown in Africa. Plant diseases are major sources of poor yields. To diagnose diseases, farmers require the help of agricultural experts to visually inspect the plants, which is labor-intensive and costly. Deep learning helps to automate this process.

This project works with a dataset of 21,367 cassava images. The pictures are taken by farmers on mobile phones and labeled as healthy or having one of the 4 common disease types. Main data-related challenges are poor image quality, inconsistent background conditions and label noise.

We develop a stacking ensemble with CNNs and Vision Transformers implemented in `PyTorch`. Our solution reaches the test accuracy of 91.06% and places 14th out of 3,900 competing teams. The diagram below overviews the ensemble. The detailed summary of our solution is provided [this writeup](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/220751).

![cassava](https://i.postimg.cc/d1dcZ6Zv/cassava.png)


## Project structure

The project has the following structure:
- `codes/`: `.py` scripts with training, inference and data processing functions
- `notebooks/`: `.ipynb` notebooks for data eploration, training CNN/ViT models and ensembling
- `data/`: input data (images are not included due to size constraints and can be downloaded [here](https://www.kaggle.com/c/cassava-leaf-disease-classification))
- `output/`: model configurations, weights and diagrams exported from notebooks
- `pretraining/`: model configurations and weights pretrained on external datasets


## Working with the repo

### Environment

To execute codes, you can create a virtual Conda environment from the `environment.yml` file:
```
conda env create --name cassava --file environment.yml
conda activate cassava
```

### Reproducing solution

Our solution can be reproduced in the following steps:
1. Downloading competition data and adding it into the `data/` folder.
2. Running training notebooks `pytorch-model` to obtain base models weights.
3. Running the ensembling notebook `lightgbm-stacking` to get final predictions.

All `pytorch-model` notebooks have the same structure and differ in model/data parameters. Different versions are included to ensure reproducibility. If you only wish to get familiar with our solution, it is enough to inspect one of the PyTorch modeling codes and go through the `functions/` folder to understand the training process. The stacking ensemble reproducing our submission is also provided in this [Kaggle notebook](https://www.kaggle.com/kozodoi/14th-place-solution-stack-them-all).

The notebooks are designed to run on Google Colab. More details are provided in the documentation within the notebooks.
