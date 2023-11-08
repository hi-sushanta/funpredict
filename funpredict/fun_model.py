import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import time
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from funpredict.utils import Utils
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
import warnings

ut = Utils()

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)


class PlayClassifier:
    """
    PlayClassifier is a game-changing tool that makes comparing and contrasting classification 
    algorithms in Scikit-learn a breeze. Its easy-to-use interface and a range of customization 
    options enable you to quickly and efficiently assess your data to determine the ideal algorithm 
    for your particular needs.
    ----------
    * `verbose`: Optional, sets verbosity level (default 0). If set to a positive value, 
            the liblinear and lbfgs solvers will be more communicative during training.

    * `ignore_warnings`: Defaulted to True, this will prevent warnings from algorithms that can't run from appearing.
    
    * `custom_metric`: An optional function that can be provided to evaluate models according to a custom evaluation metric.
   
    * `prediction`: Defaulted to False, if set to True, the predictions of all the models will be returned as a dataframe.
    
    * `classifiers`: An optional list of classifiers to train (defaults to "all"). If a function is provided, the chosen classifiers will be trained using that function.

    Examples
    --------
    >>> from funpredict.fun_model import PlayClassifier
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_wine()
    >>> X,y = data.data ,data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =42)
    >>> clf = PlayClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    >>> model,pred = clf.fit(X_train, X_test, y_train, y_test,task='multiclass')
    >>> model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test)
    >>> model
                                        | Accuracy | Balanced Accuracy| F1 Score | Time Taken |
                                        ---------------------------------------------------
      Model :                           
                                        -------------------------------------------------------
    | ExtraTreesClassifier              | 1.00     |  1.00            |   1.00    | 0.27      |
    | RandomForestClassifier            | 1.00     |  1.00            |   1.00    | 0.40      |
    | GaussianNB                        | 1.00     |  1.00            |   1.00    | 0.02      |
    | CatBoostClassifier                | 0.99     |  0.99            |   0.99    | 3.32      |
    | KNeighborsClassifier              | 0.99     |  0.99            |   0.99    | 0.03      |
    | RidgeClassifierCV                 | 0.99     |  0.99            |   0.99    | 0.02      |
    | PassiveAggressiveClassifier       | 0.99     |  0.99            |   0.99    | 0.04      |
    | LogisticRegression                | 0.99     |  0.99            |   0.99    | 0.03      |
    | NearestCentroid                   | 0.98     |  0.98            |   0.98    | 0.03      |
    | LGBMClassifier                    | 0.98     |  0.98            |   0.98    | 0.15      |
    | Perceptron                        | 0.98     |  0.98            |   0.98    | 0.04      |
    | SGDClassifier                     | 0.98     |  0.98            |   0.98    | 0.02      |
    | LinearDiscriminantAnalysis        | 0.98     |  0.98            |   0.98    | 0.02      |
    | LinearSVC                         | 0.98     |  0.98            |   0.98    | 0.02      |
    | RidgeClassifier                   | 0.98     |  0.98            |   0.98    | 0.02      |
    | NuSVC                             | 0.98     |  0.98            |   0.98    | 0.02      |
    | SVC                               | 0.98     |  0.98            |   0.98    | 0.02      |
    | LabelPropagation                  | 0.97     |  0.97            |   0.97    | 0.02      |
    | LabelSpreading                    | 0.97     |  0.97            |   0.97    | 0.02      |
    | XGBClassifier                     | 0.97     |  0.97            |   0.97    | 0.10      |
    | BaggingClassifier                 | 0.97     |  0.97            |   0.97    | 0.11      |
    | BernoulliNB                       | 0.94     |  0.94            |   0.94    | 0.04      |
    | CalibratedClassifierCV            | 0.94     |  0.94            |   0.94    | 0.15      |
    | AdaBoostClassifier                | 0.93     |  0.93            |   0.93    | 0.29      |
    | QuadraticDiscriminantAnalysis     | 0.93     |  0.93            |   0.93    | 0.04      |
    | DecisionTreeClassifier            | 0.88     |  0.88            |   0.88    | 0.04      |
    | ExtraTreeClassifier               | 0.83     |  0.83            |   0.83    | 0.04      |
    | DummyClassifier                   | 0.34     |  0.33            |   0.17    | 0.03      |

    """
    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        classifiers="all",
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.classifiers = classifiers

    def fit(self, X_train, X_test, y_train, y_test,task='binary'):
        """Hey there, data aficionado! This function is your secret weapon for finding the best-performing 
           model for your classification task. It trains and evaluates a bunch of models on your data and 
           spits out a nice report card of how each one fared. You'll get a handy table of evaluation metrics 
           and another table of predictions from each model. Now you can be confident that you've chosen the 
           right model for the job, and have some awesome insights into your data to boot! 🤓

        Args:
            X_train(`np.ndarray`): Training data features.
            X_test(`np.ndarray`): Testing data features.
            y_train(`np.ndarray`): Training data labels.
            y_test(`np.ndarray`): Testing data labels.

        Returns:
            scores (`Pandas DataFrame`): A DataFrame containing the evaluation metrics for all models.
            predictions (`Pandas DataFrame`): A DataFrame containing the predictions for all models.
        """
        Accuracy = []
        B_Accuracy = []
        ROC_AUC = []
        F1 = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric is not None:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = ut.get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", ut.numeric_transformer, numeric_features),
                ("categorical_low", ut.categorical_transformer_low, categorical_low),
                ("categorical_high", ut.categorical_transformer_high, categorical_high),
            ]
        )

        if self.classifiers == "all":
            self.classifiers = ut.CLASSIFIERS
        else:
            try:
                temp_list = []
                for classifier in self.classifiers:
                    full_name = (classifier.__name__, classifier)
                    temp_list.append(full_name)
                self.classifiers = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Classifier(s)")

        for name, model in tqdm(self.classifiers):
            start = time.time()
            try:
                if "random_state" in model().get_params().keys():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("classifier", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("classifier", model())]
                    )

                pipe.fit(X_train, y_train)
                self.models[name] = pipe
                y_pred = pipe.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                try:
                    roc_auc = roc_auc_score(y_test, y_pred)
                except Exception as exception:
                    roc_auc = None
                    if self.ignore_warnings is False:
                        print("ROC AUC couldn't be calculated for " + name)
                        print(exception)
                names.append(name)
                Accuracy.append(accuracy)
                B_Accuracy.append(b_accuracy)
                ROC_AUC.append(roc_auc)
                F1.append(f1)
                TIME.append(time.time() - start)
                if self.custom_metric is not None:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)
                if self.verbose > 0:
                    if (self.custom_metric is not None):
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                self.custom_metric.__name__: custom_metric,
                                "Time taken": time.time() - start,
                            }
                        )
                   
                    else:
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                "Time taken": time.time() - start,
                            }
                        )
                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)
        if self.custom_metric is None and (task == 'binary'):
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    "Time Taken": TIME,
                }
            )
        elif self.custom_metric is None and (task == "multiclass"):
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "F1 Score": F1,
                    "Time Taken": TIME,
                }
            )
        else:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    self.custom_metric.__name__: CUSTOM_METRIC,
                    "Time Taken": TIME,
                }
            )
        scores = scores.sort_values(by="Balanced Accuracy", ascending=False).set_index(
            "Model"
        )

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """
       Get all trained machine learning models.If the models have not been trained yet, 
       this function will train them and than return model.
       
       Args:
           X_train(`np.ndarray`): Training data features.
           X_test(`np.ndarray`): Testing data features.
           y_train(`np.ndarray`): Training data labels.
           y_test(`np.ndarray`): Testing data labels.

        Returns:
           models (`dict`): A dictionary containing all trained machine learning models, with the model name as the key and the model pipeline as the value.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models
    
    def barplot(self,pred,score_label='Accuracy',backcolor='violet',font_size=16,title_color="black",ac= "green",tc='yellow'):
        ut.cbarplot(pred,score_label,backcolor,font_size,title_color,ac,tc)
    def hbarplot(self,pred,score_label='Accuracy',backcolor='violet',font_size=16,title_color='black',ac='green',tc='yellow'):
        ut.chbarplot(pred,score_label,backcolor,font_size,title_color,ac,tc)





class PlayRegressor:
    """
    This awesome module is like a personal trainer for regression models in Scikit-learn. 
    It helps you fit and evaluate a range of regression models,so you can choose the best model 
    for your data    
    
    Args:
        verbose(`int`, *optional*, defaults to 0): The verbosity level of the model fitting process. Higher values result in more detailed output.
        ignore_warnings(`bool`, *optional*, defaults to `True`): Whether to ignore warnings related to algorithms that are not able to run.
        custom_metric(`function`, *optional*, defaults to `None`): A custom evaluation metric to use when evaluating the models.
        predictions(`bool`, *optional* defaults to `False`): Whether to return the predictions of all the models as a DataFrame.
        random_state (`int`, *optional*, defaults to 42): The random state to use for all model fitting operations.
        regressors(`list`, *optional*, defaults to `all`): A list of the Scikit-learn regressors to train. If "all", all available regressors are trained.

    Examples:
    >>> from funpredict.fun_model import PlayRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_diabetes()
    >>> X,y = data.data, data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =42)

    >>> clf = PlayRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
    >>> models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test) # If you conform which model working best than choosen hare.
    >>> models

    | Model                         | Adjusted R-Squared | R-Squared |  RMSE | Time Taken |
    |:------------------------------|-------------------:|----------:|------:|-----------:|
    | BayesianRidge                 |      0.45          |   0.48    | 54.46 |    0.04    |
    | ElasticNetCV                  |      0.46          |   0.48    | 54.41 |    0.31    |
    | RidgeCV                       |      0.45          |   0.48    | 54.51 |    0.04    |
    | LinearRegression              |      0.45          |   0.48    | 54.58 |    0.03    |
    | TransformedTargetRegressor    |      0.45          |   0.48    | 54.58 |    0.04    |
    | Lars                          |      0.45          |   0.48    | 54.58 |    0.05    |
    | Ridge                         |      0.45          |   0.48    | 54.59 |    0.03    |
    | Lasso                         |      0.45          |   0.47    | 54.69 |    0.03    |
    | LassoLars                     |      0.45          |   0.47    | 54.69 |    0.03    |
    | LassoCV                       |      0.45          |   0.47    | 54.70 |    0.28    |
    | LassoLarsCV                   |      0.45          |   0.47    | 54.71 |    0.07    |
    | PoissonRegressor              |      0.45          |   0.47    | 54.76 |    0.04    |
    | SGDRegressor                  |      0.45          |   0.47    | 54.76 |    0.04    |
    | OrthogonalMatchingPursuitCV   |      0.45          |   0.47    | 54.80 |    0.06    |
    | HuberRegressor                |      0.44          |   0.47    | 54.96 |    0.06    |
    | LassoLarsIC                   |      0.44          |   0.47    | 55.02 |    0.03    |
    | ElasticNet                    |      0.44          |   0.47    | 55.05 |    0.03    |
    | LarsCV                        |      0.43          |   0.45    | 55.72 |    0.09    |
    | AdaBoostRegressor             |      0.42          |   0.44    | 56.34 |    0.34    |
    | TweedieRegressor              |      0.41          |   0.44    | 56.40 |    0.03    |
    | ExtraTreesRegressor           |      0.41          |   0.44    | 56.60 |    0.40    |
    | PassiveAggressiveRegressor    |      0.41          |   0.44    | 56.61 |    0.03    |
    | GammaRegressor                |      0.41          |   0.43    | 56.79 |    0.02    |
    | LGBMRegressor                 |      0.40          |   0.43    | 57.04 |    0.12    |
    | CatBoostRegressor             |      0.39          |   0.42    | 57.47 |    3.26    |
    | RandomForestRegressor         |      0.38          |   0.41    | 58.00 |    0.79    |
    | HistGradientBoostingRegressor |      0.36          |   0.39    | 58.84 |    0.27    |
    | GradientBoostingRegressor     |      0.36          |   0.39    | 58.95 |    0.31    |
    | BaggingRegressor              |      0.33          |   0.36    | 60.12 |    0.11    |
    | KNeighborsRegressor           |      0.29          |   0.32    | 62.09 |    0.03    |
    | XGBRegressor                  |      0.23          |   0.27    | 64.59 |    0.21    |
    | OrthogonalMatchingPursuit     |      0.23          |   0.26    | 64.86 |    0.05    |
    | RANSACRegressor               |      0.11          |   0.15    | 69.40 |    0.33    |
    | NuSVR                         |      0.07          |   0.11    | 70.99 |    0.08    |
    | LinearSVR                     |      0.07          |   0.11    | 71.11 |    0.03    |
    | SVR                           |      0.07          |   0.11    | 71.23 |    0.04    |
    | DummyRegressor                |      0.05      -   |   0.00    | 75.45 |    0.02    |
    | DecisionTreeRegressor         |      0.13      -   |   0.08    | 78.38 |    0.03    |
    | ExtraTreeRegressor            |      0.18      -   |   0.13    | 80.02 |    0.02    |
    | GaussianProcessRegressor      |      0.99      -   |   0.90    | 04.06 |    0.07    |
    | MLPRegressor                  |      1.19      -   |   1.09    | 09.17 |    1.34    |
    | KernelRidge                   |      3.91      -   |   3.69    | 63.34 |    0.06    |
    |______________________________________________________________________________________ 
    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        regressors="all",
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.regressors = regressors

    def fit(self, X_train, X_test, y_train, y_test):
        """Fit the mathematical magic of regression algorithms to the glorious X_train and y_train, 
           and with their powers combined, predict and score like a boss on X_test and y_test!" 👊

        Args: 
            X_train(`np.ndarray`): Training data, where rows are samples and columns are features.
            X_test(`np.ndarray`): Test data, where rows are samples and columns are features.
            y_train(`np.ndarray`): Training labels.
            y_test(`np.ndarray`): Testing labels.
        
        Returns:
               scores(`Pandas DataFrame`): A Pandas DataFrame containing the evaluation metrics of all the models.
               predictions(`Pandas DataFrame`): A Pandas DataFrame containing the evaluation metrics of all the models.
        """
        R2 = []
        ADJR2 = []
        RMSE = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = ut.get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", ut.numeric_transformer, numeric_features),
                ("categorical_low", ut.categorical_transformer_low, categorical_low),
                ("categorical_high", ut.categorical_transformer_high, categorical_high),
            ]
        )

        if self.regressors == "all":
            self.regressors = ut.REGRESSORS
        else:
            try:
                temp_list = []
                for regressor in self.regressors:
                    full_name = (regressor.__name__, regressor)
                    temp_list.append(full_name)
                self.regressors = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Regressor(s)")

        for name, model in tqdm(self.regressors):
            start = time.time()
            try:
                if "random_state" in model().get_params().keys():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("regressor", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("regressor", model())]
                    )

                pipe.fit(X_train, y_train)
                self.models[name] = pipe
                y_pred = pipe.predict(X_test)

                r_squared = r2_score(y_test, y_pred)
                adj_rsquared = ut.adjusted_rsquared(
                    r_squared, X_test.shape[0], X_test.shape[1]
                )
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                names.append(name)
                R2.append(r_squared)
                ADJR2.append(adj_rsquared)
                RMSE.append(rmse)
                TIME.append(time.time() - start)

                if self.custom_metric:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)

                if self.verbose > 0:
                    scores_verbose = {
                        "Model": name,
                        "R-Squared": r_squared,
                        "Adjusted R-Squared": adj_rsquared,
                        "RMSE": rmse,
                        "Time taken": time.time() - start,
                    }

                    if self.custom_metric:
                        scores_verbose[self.custom_metric.__name__] = custom_metric

                    print(scores_verbose)
                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)

        scores = {
            "Model": names,
            "Adjusted R-Squared": ADJR2,
            "R-Squared": R2,
            "RMSE": RMSE,
            "Time Taken": TIME,
        }

        if self.custom_metric:
            scores[self.custom_metric.__name__] = CUSTOM_METRIC

        scores = pd.DataFrame(scores)
        scores = scores.sort_values(by="Adjusted R-Squared", ascending=False).set_index(
            "Model"
        )

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        This fine function grabs all the trained models created in the fit function and serves 'em up. 
        If fit hasn't been called yet, this function gets the party started,trains those bad boys up and 
        then sends 'em back like a boss."

        Args:
            X_train(`np.ndarray`): Training data, where rows are samples and columns are features.
            X_test(`np.ndarray`): Test data, where rows are samples and columns are features.
            y_train(`np.ndarray`): Training labels.
            y_test(`np.ndarray`): Test labels.
        
        Returns:
            models(`dict`): Returns a dictionary of fitted models, with the model name as the key
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models
    
    def barplot(self,pred,score_label ="Adjusted R-Squared",backcolor='violet',font_size=16,title_color="black",ac= "green",tc='yellow'):
        ut.rbarplot(pred,score_label,backcolor,font_size,title_color,ac,tc)
    def hbarplot(self,pred,score_label="Adjusted R-Squared",backcolor='violet',font_size=16,title_color='black',ac='green',tc='yellow'):
        ut.rhbarplot(pred,score_label,backcolor,font_size,title_color,ac,tc)