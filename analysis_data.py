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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, average_precision_score

scoring = {'acurracy' : make_scorer(accuracy_score), 
               'precision' : make_scorer(precision_score),
               'rollup' : make_scorer(recall_score)}
               
OUTPUT_TEMPLATE = (
    'Bayesian classifier:    {bayes:.3f} \n'
    'kNN classifier:         {knn:.3f} \n'
    'rf classifier:          {rf:.3f} \n'
    'dt classifier:          {dt:.3f} \n'
    'mlp classifier:         {mlp:.3f} \n'
    'sv classifier:         {svc:.3f} \n'
    
)

def main():

    data = pd.read_csv("features_amag.csv")

    X = data.drop('has_injury', axis=1).values
    y = data['has_injury'].values

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
        DecisionTreeClassifier(max_depth=5, )
    )

    mlp_model = make_pipeline(
        MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4,3), activation='logistic', max_iter=1000)
    )

    svc_model = make_pipeline(
        SVC()
    )

    scores = pd.DataFrame(columns=['bayes_model', 'knn_model', 'rf_model', 'dt_model', 'mlp_model', 'svc_model'])
    models = [bayes_model, knn_model, rf_model, dt_model, mlp_model, svc_model]
    for k in range(20):
        for i, m in enumerate(models):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.4)
            m.fit(X_train, y_train)

        row = [bayes_model.score(X_valid, y_valid),
                knn_model.score(X_valid, y_valid),
                rf_model.score(X_valid, y_valid),
                dt_model.score(X_valid, y_valid),
                mlp_model.score(X_valid, y_valid),
                svc_model.score(X_valid, y_valid)]
        scores.loc[k] = row

    print(OUTPUT_TEMPLATE.format(
            bayes=scores['bayes_model'].mean(),
            knn=scores['knn_model'].mean(),
            rf=scores['rf_model'].mean(),
            dt=scores['dt_model'].mean(),
            mlp=scores['mlp_model'].mean(),
            svc=scores['svc_model'].mean()
        ))


if __name__=='__main__':
    main()