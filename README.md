# Kaggle: Cassava Leaf Disease Classification Solution

Files and codes with the 14th place solution to the [Cassava Leaf Disease Classification Kaggle competition](https://www.kaggle.com/c/cassava-leaf-disease-classification).


## Summary

Cassava is one of the key food crops grown by farmers in Africa. Plant diseases are major sources of poor yields. To diagnose diseases, farmers require the help of agricultural experts to visually inspect the plants, which is labor-intensive and costly. Deep learning helps to automate this process.

This project works with a dataset of 21,367 cassava images. The pictures are taken by farmers on mobile phones and labeled as healthy or having one of the 4 common disease types. Main data-related challenges are poor image quality, inconsistent background conditions and label noise.

We develop a stacking ensemble with CNNs and Vision Transformers implemented in `PyTorch`. Our solution reaches the test accuracy of 91.06% and places 14th out of 3,900 competing teams. The diagram below overviews the ensemble. The detailed summary of our solution is provided [this writeup](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/220751).

![cassava](https://i.postimg.cc/d1dcZ6Zv/cassava.png)


## Project structure

The project has the following structure:
- `notebooks/`: `.ipynb` notebooks performing training of CNN/ViT models and ensembling.
- `functions/`: `.py` modules supporting the notebooks including training, inference and data processing.
- `data/`: input data. Images are not uploaded to GitHub due to size constraints and can be downloaded [here](https://www.kaggle.com/c/cassava-leaf-disease-classification).
- `output/`: model weights and diagrams exported from notebooks.
- `pretraining/`: model weights pretrained on external datasets.


## Modeling pipeline

Our solution can be reproduced in the following steps:
1. Running all training notebooks to obtain weights of 33+2 base models for the ensemble.
2. Running the ensembling notebook to obtain the final prediction.

The notebooks are designed to run on Google Colab and require downloading the competition data to the corresponding folders. More details are provided in the documentation within the notebooks.
