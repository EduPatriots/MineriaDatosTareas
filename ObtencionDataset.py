from zipfile import ZipFile
import pandas as pd
import os

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

api.dataset_download_files('tobycrabtree/nfl-scores-and-betting-data')

zf = ZipFile('nfl-scores-and-betting-data.zip')
#extracted data is saved in the same directory as notebook
zf.extractall() 
zf.close()

os.remove('spreadspoke.R')
os.remove('nfl-scores-and-betting-data.zip')

data=pd.read_csv('spreadspoke_scores.csv')
print(data)