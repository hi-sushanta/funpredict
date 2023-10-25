from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline
import pandas as pd

import xgboost

import catboost
import lightgbm
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder



class Utils:

    def __init__(self):
       self.r_clf = [
            "ClassifierChain",
            "ComplementNB",
            "GradientBoostingClassifier",
            "GaussianProcessClassifier",
            "HistGradientBoostingClassifier",
            "MLPClassifier",
            "LogisticRegressionCV", 
            "MultiOutputClassifier", 
            "MultinomialNB", 
            "OneVsOneClassifier",
            "OneVsRestClassifier",
            "OutputCodeClassifier",
            "RadiusNeighborsClassifier",
            "VotingClassifier",
        ] 
       self.r_rgs = [
            "TheilSenRegressor",
            "ARDRegression", 
            "CCA", 
            "IsotonicRegression", 
            "StackingRegressor",
            "MultiOutputRegressor", 
            "MultiTaskElasticNet", 
            "MultiTaskElasticNetCV", 
            "MultiTaskLasso", 
            "MultiTaskLassoCV", 
            "PLSCanonical", 
            "PLSRegression", 
            "RadiusNeighborsRegressor", 
            "RegressorChain", 
            "VotingRegressor", 
        ]
       self.CLASSIFIERS = [
                    est
                    for est in all_estimators()
                            if (issubclass(est[1], ClassifierMixin) and (est[0] not in self.r_clf))
                    ]  
       self.CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
       self.CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
       self.CLASSIFIERS.append(('CatBoostClassifier',catboost.CatBoostClassifier))


       self.REGRESSORS = [
                est
                for est in all_estimators()
                if (issubclass(est[1], RegressorMixin) and (est[0] not in self.r_rgs))
            ]

       self.REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
       self.REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
       self.REGRESSORS.append(('CatBoostRegressor',catboost.CatBoostRegressor))

       self.numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
            )
       self.categorical_transformer_low = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                ]
            )  
       self.categorical_transformer_high = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoding", OrdinalEncoder()),
                ]
            )
    def get_card_split(self,df, cols, n=11):
        """
        Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)
        Parameters
        ----------
        df : Pandas DataFrame
            DataFrame from which the cardinality of the columns is calculated.
        cols : list-like
            Categorical columns to list
        n : int, optional (default=11)
            The value of 'n' will be used to split columns.
        Returns
        -------
        card_low : list-like
            Columns with cardinality < n
        card_high : list-like
            Columns with cardinality >= n
        """
        cond = df[cols].nunique() > n
        card_high = cols[cond]
        card_low = cols[~cond]
        return card_low, card_high
    
    def adjusted_rsquared(self,r2, n, p):
        return 1 - (1 - r2) * ((n - 1) / (n - p - 1))



