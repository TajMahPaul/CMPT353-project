import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

OUTPUT_TEMPLATE = (
    'Bayesian classifier:    {bayes_rgb:.3f} \n'
    'kNN classifier:         {knn_rgb:.3f} \n'
)

def create_test_df():
    myList = []

    column_name = ['R', 'G', 'B', 'Label']
    a = [168, 211, 243, 'blue']
    b = [37, 32, 40, 'black']
    c = [35, 34, 38, 'black']
    d = [79,74 ,159, 'purple']
    e = [55, 99, 34, 'green']

    myList.append(a)
    myList.append(b)
    myList.append(c)
    myList.append(d)
    myList.append(e)

    df = pd.DataFrame(data=myList, columns=column_name)
    return df
    

def main():
    data = create_test_df()
    print(data)

    X = data[['R', 'G', 'B']].values / 255
    y = data['Label'].values

    print(X)
    print(y)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    bayes_rgb_model = make_pipeline(
        GaussianNB()
    )

    knn_rgb_model = make_pipeline(
        KNeighborsClassifier(n_neighbors=2)
    )

    models = [bayes_rgb_model, knn_rgb_model]
    for i, m in enumerate(models):  
        m.fit(X_train, y_train)

    print(OUTPUT_TEMPLATE.format(
            bayes_rgb=bayes_rgb_model.score(X_valid, y_valid),
            knn_rgb=knn_rgb_model.score(X_valid, y_valid),
        ))


if __name__=='__main__':
    main()