# Data

The dataset is not stored in this repository. On the first run, the notebook and `src/cleaning.py` download a public copy of the Kaggle Titanic training set (891 passengers) and save it here as `data/titanic.csv`.

If you'd rather download it yourself, get `train.csv` from [Kaggle](https://www.kaggle.com/c/titanic/data) and save it here as `titanic.csv`.

Running the pipeline writes the cleaned output to `data/titanic_clean.csv`.
