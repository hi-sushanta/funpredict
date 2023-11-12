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
import matplotlib.pyplot as plt
import pandas as pd


class Utils:

    def __init__(self):
       self.sclf = {"RidgeClassifierCV" :"RCCV","LogisticRegression":"LGR" ,"CalibratedClassifierCV":"CCV",             
        "SGDClassifier":"SGDC","ExtraTreesClassifier":"EXTC","GaussianNB":"GSB","RidgeClassifier":"RC",
        "PassiveAggressiveClassifier":"PAC","LinearSVC":"LSVC","RandomForestClassifier":"RFC",
        "SVC":"SVC","NuSVC":"NuSVC","NearestCentroid":"NC","LinearDiscriminantAnalysis":"LDA",
        "BernoulliNB":"BlB","CatBoostClassifier":"CBC","Perceptron":"PC","KNeighborsClassifier":"KNC",
        "LabelSpreading":"LS","LGBMClassifier" :"LGBMC","LabelPropagation":"LP","BaggingClassifier":"BC",                  
        "QuadraticDiscriminantAnalysis" : "QDA", "DecisionTreeClassifier" : "DTC", "XGBClassifier" : "XGBC",                     
        "AdaBoostClassifier": "ABC" ,"ExtraTreeClassifier" :"EXTC" ,"DummyClassifier": "DC"
        }
       
       self.rmd = {"ElasticNetCV":"ENCV","BayesianRidge":"ByesR","RidgeCV":"RCV","LinearRegression":"LR",
        "TransformedTargetRegressor":"TTR","Lars":"Lrs","Ridge":"Rige","Lasso":"Laso","LassoLars":"LL",
        "LassoCV":"LsCV","LassoLarsCV":"LLCV","PoissonRegressor":"PR","SGDRegressor":"SGDR",
        "OrthogonalMatchingPursuitCV":"OMCV","HuberRegressor":"HR","LassoLarsIC":"LLIC","ElasticNet":"ENet",
        "LarsCV":"LCV","AdaBoostRegressor":"ABR","TweedieRegressor":"TR","ExtraTreesRegressor":"ExTR",
        "PassiveAggressiveRegressor":"PAR","GammaRegressor":"GR","LGBMRegressor":"LGBMR",
        "CatBoostRegressor":"CBR","RandomForestRegressor":"RFR","HistGradientBoostingRegressor":"HGBR",
        "GradientBoostingRegressor":"GBR","BaggingRegressor":"BR","KNeighborsRegressor":"KNR",
        "XGBRegressor":"XGBR","OrthogonalMatchingPursuit":"OMP","RANSACRegressor":"RASCR",
        "NuSVR":"NSVR","LinearSVR":"LSVR","SVR":"SVR","DummyRegressor":"DR","DecisionTreeRegressor":"DTR",
        "ExtraTreeRegressor":"ETR","GaussianProcessRegressor":"GPR","MLPRegressor":"MLPR","KernelRidge":"KrR"
       }     
       


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
    
    def cbarplot(self,pred,score_label,backcolor,font_size,title_color,score_label_color,time_color):
        # Create a DataFrame from the data
        df = pd.DataFrame({'Model': self.sclf.values(),
                           score_label:pred[score_label],
                           'Time taken (seconds)': pred['Time Taken']})

        # Sort the DataFrame by accuracy
        df = df.sort_values(by=[score_label], ascending=False)

        # Create a bar chart of the accuracy vs time taken
        plt.figure(figsize=(100,100))
        plt.rcParams['axes.facecolor'] = backcolor

        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        plt.bar(df['Model'], df[score_label], color=score_label_color)
        plt.bar(df['Model'], df['Time taken (seconds)'], color=time_color)

        # Add a legend
        plt.legend([score_label, 'Time taken (seconds)'])
        # Set the chart title and labels
        plt.title(f'{score_label} vs time taken for machine learning models',color=title_color,fontsize=font_size)
        plt.xlabel('Model')
        plt.ylabel(f'{score_label} and time taken')

        # Display the chart
        plt.show()

    def rbarplot(self,pred,score_label,backcolor,font_size,title_color,score_label_color,time_color):
        # Create a DataFrame from the data
        df = pd.DataFrame({'Model': self.rmd.values(),
                           score_label:pred[score_label],
                           'Time taken (seconds)': pred['Time Taken']})

        # Sort the DataFrame by accuracy
        df = df.sort_values(by=[score_label], ascending=False)

        # Create a bar chart of the accuracy vs time taken
        plt.figure(figsize=(100,100))
        plt.rcParams['axes.facecolor'] = backcolor

        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        plt.bar(df['Model'], df[score_label], color=score_label_color)
        plt.bar(df['Model'], df['Time taken (seconds)'], color=time_color)

        # Add a legend
        plt.legend([score_label, 'Time taken (seconds)'])
        # Set the chart title and labels
        plt.title(f'{score_label} vs time taken for machine learning models',color=title_color,fontsize=font_size)
        plt.xlabel('Model')
        plt.ylabel(f'{score_label} and time taken')

        # Display the chart
        plt.show()
    
    def chbarplot(self,pred,score_label,backcolor,font_size,title_color,score_label_color,
                  time_color):
        # Create a DataFrame from the data
        df = pd.DataFrame({'Model': self.sclf.values(),
                           score_label:pred[score_label],
                           'Time taken (seconds)': pred['Time Taken']})

        # Sort the DataFrame by accuracy
        df = df.sort_values(by=[score_label], ascending=False)

        # Create a bar chart of the accuracy vs time taken
        plt.figure(figsize=(100,100))
        plt.rcParams['axes.facecolor'] = backcolor

        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        plt.barh(df['Model'], df[score_label], color=score_label_color)
        plt.barh(df['Model'], df['Time taken (seconds)'], color=time_color)

        # Add a legend
        plt.legend([score_label, 'Time taken (seconds)'])
        
        # Set the chart title and labels
        plt.title(f'{score_label} vs time taken for machine learning models',color=title_color,fontsize=font_size)
        plt.xlabel('Model')
        plt.ylabel(f'{score_label} and time taken')

        # Display the chart
        plt.show()
    
    def rhbarplot(self,pred,score_label,backcolor,font_size,title_color,score_label_color,
                  time_color):
        
        # Create a DataFrame from the data
        df = pd.DataFrame({'Model': self.rmd.values(),
                           score_label:pred[score_label],
                           'Time taken (seconds)': pred['Time Taken']})

        # Sort the DataFrame by accuracy
        df = df.sort_values(by=[score_label], ascending=False)

        # Create a bar chart of the accuracy vs time taken
        plt.figure(figsize=(100,100))
        plt.rcParams['axes.facecolor'] = backcolor
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        plt.barh(df['Model'], df[score_label], color=score_label_color)
        plt.barh(df['Model'], df['Time taken (seconds)'], color=time_color)

        # Add a legend
        plt.legend([score_label, 'Time taken (seconds)'])

        # Set the chart title and labels
        plt.title(f'{score_label} vs time taken for machine learning models',color=title_color,fontsize=font_size)
        plt.xlabel('Model')
        plt.ylabel(f'{score_label} and time taken')

        # Display the chart
        plt.show()




