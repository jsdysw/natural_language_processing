import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data(dataset_path):
    data = load_dataset(dataset_path)
    X, y = preprocess(data)
    return split_dataset(X, y)


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,test_size=0.05)

    y_train = [1 if y == 'positive' else 0 for y in y_train]
    y_valid = [1 if y == 'positive' else 0 for y in y_valid]
    y_test =  [1 if y == 'positive' else 0 for y in y_test]

    return X_train, y_train, X_test, y_test, X_valid, y_valid


def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)


def preprocess(data):
    data['review'].nunique(), data['sentiment'].nunique()
    data.drop_duplicates(subset=['review'], inplace=True) # remove duplicates in document column

    if data.isnull().values.any():
        # data.loc[data.document.isnull()]
        data = data.dropna(how = 'any') # remove Null columns

    # data['sentiment'] = data['sentiment'].replace(['negative','positive'], [0,1])
    
    data['review'] = data['review'].str.lower()
    data['review'] = data['review'].str.replace('<[^>]*>','', regex=True)
    data['review'] = data['review'].str.replace(r'[^a-zA-Z ]','', regex=True)
    data['review'] = data['review'].str.replace('^ +', '', regex=True) # white space -> empty value
    data['review'].replace('', np.nan, inplace=True) # '' -> null

    if data.isnull().values.any():
        # print(data.loc[data.review.isnull()])
        data = data.dropna(how = 'any') # remove Null columns

    y = data.sentiment
    X = data.drop('sentiment', axis=1)
    return X, y