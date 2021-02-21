# Kaggle Cassava Leaf Disease Classification

Files and codes with 14th place solution to the [Kaggle Cassava Leaf Disease Classification competition](https://www.kaggle.com/c/cassava-leaf-disease-classification).


## Project summary

Cassava is one of the key food crops grown by farmers in Africa. Plant diseases are major sources of poor yields. In order to diagnose diseases, farmers require the help of agricultural experts to visually inspect the plants, which is labor-intensive and costly. Deep learning may help to automate this process.

This project works with a dataset of 21,367 cassava images. The pictures are taken by different farmers on mobile phones and labeled as healthy or one of the 4 common disease types. Main data-related challenges are poor image quality, inconsistent background conditions and label noise.

We develop a stacking ensemble of different CNN and Vision Transformer models that reaches the accuracy of 91.06% on the hidden test set. See the diagram below for the ensemble overview. The coding is done in `PyTorch`. The detailed summary of our solution is provided [this writeup](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/220751).
![cassava](https://i.postimg.cc/d1dcZ6Zv/cassava.png)


## Project structure

The project has the following structure:
- `notebooks/`: notebooks performing the training of base models and ensembling. The notebooks are Colab-friendly.
- `functions/`: utilites supporting the training notebooks including training, inference and data processing
- `input/`: input data. The images are not uploaded to Github due to size constraints. The raw data can be downloaded [here](https://www.kaggle.com/c/cassava-leaf-disease-classification).
- `output/`: model weights and figures exported from notebooks.
- `pretraining/`: model weights pretrained on external datasets.

## Modeling pipeline

Our solution can be reproduced in the following steps:
1. Running all training notebooks to obtain weights of 33+2 base models for the ensemble.
2. Running the ensembling notebook to obtain the final prediction.

More specific details are provided in the documentation within the notebooks.
