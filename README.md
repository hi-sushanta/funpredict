# Fun Predict🤖
Fun Predict is a free, open-source Python library that helps you build and compare machine learning models easily, without writing much code. It allows you to quickly and easily evaluate a variety of models without having to write a lot of code or tune hyperparameters.

# Installation
To install Fun Predict:
``` 
pip install funpredict
```

# Usage
To use Fun Predict in a project:
```
import funpredict
```

# Classification
Example :
```
from funpredict.fun_model import PlayClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Test with a Classification model
data = load_wine()
X,y = data.data,data.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =42)

clf = PlayClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test,'multiclass')
# If you confirm which model working best then choose hare.
model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test) 
print(models)

                                        | Accuracy | Balanced Accuracy| F1 Score | Time Taken |
     -----------------------------------------------------------------------------------------|
    | Model :                                                                                 |
    |                                    -----------------------------------------------------+
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
    -------------------------------------------------------------------------------------------
```
```
# Vertical bar plot
clf.barplot(predictions)
```
![clf-bar](https://github.com/hi-sushanta/funpredict/assets/93595990/6a6ab9fc-ceb1-481e-a73c-1314c59c4562)

```
# Horizontal bar plot
clf.hbarplot(predictions)
```
![clf-hbar](https://github.com/hi-sushanta/funpredict/assets/93595990/9b594717-411e-4295-90bd-37cd4ef9e68e)

# Regression
Example :
```
from funpredict.fun_model import PlayRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Test with Regressor Model
data = load_diabetes()
X,y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =42)

rgs = PlayRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = rgs.fit(X_train, X_test, y_train, y_test)
# If you confirm which model works best then choose hare.
model_dictionary = rgs.provide_models(X_train, X_test,y_train,y_test)
print(models)

|-----------------------------------------------------------------------------------------|
| Model                             | Adjusted R-Squared | R-Squared |  RMSE | Time Taken | 
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
    |-------------------------------------------------------------------------------------|
```

![rgs_all](https://github.com/hi-sushanta/funpredict/assets/93595990/f5dc96f5-ed8a-4fa4-a338-755f3b690d8f)

