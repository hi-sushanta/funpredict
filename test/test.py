from funpredict.fun_model import PlayClassifier,PlayRegressor
from sklearn.datasets import load_wine,load_diabetes
from sklearn.model_selection import train_test_split

data = load_wine()
X,y = data.data,data.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =42)

clf = PlayClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test,'multiclass')
model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test) # If you conform which model working best than choosen hare.
print(models)

# Regressor Model Train
data = load_diabetes()
X,y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =42)

clf = PlayRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test) # If you conform which model working best than choosen hare.
print(models)

