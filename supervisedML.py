# Imports

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Class

class supervisedML():

    def __init__(self, df, features, target, df_pred):
        self.df = df
        self.features = features
        self.target = target
        self.df_pred = df_pred
        self.alpha_range = alpha_range = np.arange(0.0001, 100, .1)
        self.neighbors = np.arange(1, 26)
        self.max_depth = np.arange(1, 30)
        self.random_state = 42

    def scalePredictionInput(self):
        scaler = StandardScaler()
        return scaler.fit_transform(self.df_pred[self.features].values)

    def scaleData(self):
        X = self.df[self.features].values
        y = self.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def lassoRegression(self):

        scores_dict = dict()
        X_train_scaled, X_test_scaled, y_train, y_test = self.scaleData()

        for alpha in self.alpha_range:
            lasso = Lasso(alpha=alpha)

            lasso.fit(X_train_scaled, y_train)

            scores_dict[alpha] = lasso.score(X_test_scaled, y_test)

        scores_df = pd.DataFrame(scores_dict, index=['score']).T
        scores_df['name'] = 'lasso'
        highest_score = scores_df.loc[scores_df['score'] == scores_df['score'].max()]
        return highest_score

    def ridgeRegression(self):
        scores_dict = dict()
        X_train_scaled, X_test_scaled, y_train, y_test = self.scaleData()

        for alpha in self.alpha_range:
            ridge = Ridge(alpha=alpha)

            ridge.fit(X_train_scaled, y_train)

            scores_dict[alpha] = ridge.score(X_test_scaled, y_test)

        scores_df = pd.DataFrame(scores_dict, index=['score']).T
        scores_df['name'] = 'ridge'
        highest_score = scores_df.loc[scores_df['score'] == scores_df['score'].max()]

        return highest_score

    def linearregressionModel(self):
        regr = LinearRegression()
        X_train_scaled, X_test_scaled, y_train, y_test = self.scaleData()

        regr.fit(X_train_scaled, y_train)

        prediction = regr.predict(X_test_scaled)

        score = regr.score(X_test_scaled, y_test)
        scores_df = pd.DataFrame({'score': score, 'name': 'linear'}, index=[1])
        scores_df.reset_index(0)
        return scores_df

    def knn(self):
        X_train_scaled, X_test_scaled, y_train, y_test = self.scaleData()
        scores_dict = dict()

        for neighbor in self.neighbors:
            knn = KNeighborsClassifier(n_neighbors=neighbor)
            knn.fit(X_train_scaled, y_train)
            scores_dict[neighbor] = knn.score(X_test_scaled, y_test)

        scores_df = pd.DataFrame(scores_dict, index=['score']).T
        scores_df['name'] = 'knn'
        highest_score = scores_df.loc[scores_df['score'] == scores_df['score'].max()].head(1)

        return highest_score

    def decisionTree(self):
        X_train_scaled, X_test_scaled, y_train, y_test = self.scaleData()
        scores_dict = dict()

        for depth in self.max_depth:
            clf = DecisionTreeClassifier(max_depth=depth, random_state=self.random_state)
            clf.fit(X_train_scaled, y_train)
            scores_dict[depth] = clf.score(X_test_scaled, y_test)

        scores_df = pd.DataFrame(scores_dict, index=['score']).T
        scores_df['name'] = 'decisionTree'
        highest_score = scores_df.loc[scores_df['score'] == scores_df['score'].max()].head(1)

        return highest_score

    def getBestModel(self):
        score_df = pd.DataFrame()

        lasso = self.lassoRegression()
        ridge = self.ridgeRegression()
        linearreg = self.linearregressionModel()
        knn = self.knn()
        dt = self.decisionTree()

        score_df = pd.concat([score_df, lasso, ridge, linearreg, knn, dt])

        best_model = score_df.loc[score_df['score'] == score_df['score'].max()]

        return best_model

    def makePrediction(self):
        df = self.getBestModel()
        X_train_scaled, X_test_scaled, y_train, y_test = self.scaleData()
        X_pred = self.scalePredictionInput()
        name = df['name'].values[0]

        if name == 'decisionTree':
            clf = DecisionTreeClassifier(max_depth=int(df.index[0]), random_state=self.random_state)
            clf.fit(X_train_scaled, y_train)
            predictions = clf.predict(X_pred)

        elif name == 'knn':
            knn = KNeighborsClassifier(n_neighbors=int(df.index[0]))
            knn.fit(X_train_scaled, y_train)
            predictions = knn.predict(X_pred)
        elif name == 'linear':
            regr = LinearRegression()
            regr.fit(X_train_scaled, y_train)
            predictions = regr.predict(X_pred)

        elif name == 'ridge':
            ridge = Ridge(alpha=df.index[0])
            ridge.fit(X_train_scaled, y_train)
            predictions = ridge.predict(X_pred)
        else:
            lasso = Lasso(alpha=df.index[0])
            lasso.fit(X_train_scaled, y_train)
            predictions = lasso.predict(X_pred)

        return predictions
