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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

OUTPUT_TEMPLATE = (
    'Bayesian classifier:    {bayes:.3f} \n'
    'kNN classifier:         {knn:.3f} \n'
    'rf classifier:          {rf:.3f} \n'
    'dt classifier:          {dt:.3f} \n'
    'mlp classifier:         {mlp:.3f} \n'
    'sv classifier:         {mlp:.3f} \n'
    
)

def main():

    data = pd.read_csv("features.csv")
    print(data)    


    X = data.drop('has_injury', axis=1).values
    y = data['has_injury'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    bayes_model = make_pipeline(
        GaussianNB()
    )

    knn_model = make_pipeline(
        KNeighborsClassifier(n_neighbors=3)
    )

    rf_model = make_pipeline(
        RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=2)
    )

    dt_model = make_pipeline(
        DecisionTreeClassifier(max_depth=2)
    )

    mlp_model = make_pipeline(
        MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4,4), activation='logistic')
    )

    svc_model = make_pipeline(
        SVC()
    )

    models = [bayes_model, knn_model, rf_model, dt_model, mlp_model, svc_model]
    for i, m in enumerate(models):  
        m.fit(X_train, y_train)

    print(OUTPUT_TEMPLATE.format(
            bayes=bayes_model.score(X_valid, y_valid),
            knn=knn_model.score(X_valid, y_valid),
            rf=rf_model.score(X_valid, y_valid),
            dt=dt_model.score(X_valid, y_valid),
            mlp=mlp_model.score(X_valid, y_valid),
            svc=svc_model.score(X_valid, y_valid)
        ))


if __name__=='__main__':
    main()