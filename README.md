# Kaggle Home Credit Default Risk

Files and codes with my solution to the [Kaggle Home Credit Default Risk competition](https://www.kaggle.com/c/home-credit-default-risk).


## Project summary

In finance, credit scoring refers to the use of statistical models to guide loan approval decisions. This project develops a binary classification model to distinguish defaulters and non-defaulters using supervised machine learning.

The project works with data from multiple sources, including credit bureau information, application data, performance on previous loans and credit card balance. I perform thorough feature engineering and aggregate data into a single high-dimensional data set. Next, I train an ensemble of LightGBM models that predict the probability of default.


## Project structure

The project has the following structure:
- `codes/`: notebooks with codes for three project stages: data preparation, modeling and ensembling.
- `data/`: input data. The folder is not uploaded to Github due to size constraints. The raw data can be downloaded [here](https://www.kaggle.com/c/home-credit-default-risk).
- `output/`: output figures exported from notebooks.
- `solutions/`: slides with solutions from other competitors.
- `submissions/`: predictions produced by the trained models.

There are three notebooks:
- `code_1_data_prep.ipynb`: data preprocessing and feature engineering. Processes the raw data and exports the aggregated data set.
- `code_2_modeling.ipynb`: modeling credit risk with a LightGBM model. Takes the aggregated data as input and produces submissions.
- `code_3_ensemble.ipynb`: ensembling predictions from different models.

More details are provided within the notebooks.


## Requirments

To run the project codes, you can create a new virtual environment in `conda`:

```
conda create -n py3 python=3.7
conda activate py3
```

and then install the requirements:

```
conda install -n py3 --yes --file requirements.txt
pip install lightgbm
pip install imblearn
```
