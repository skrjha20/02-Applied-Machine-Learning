import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics.regression import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
    
def part1_scatter(X_train, X_test, y_train, y_test):
    
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);

def plot_one(degree_predictions):
    
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i, degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    
def answer_one():
    
    result = np.zeros((4, 100))
    for i, degree in enumerate([1, 3, 6, 9]):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train.reshape(11,1))
        linreg = LinearRegression().fit(X_poly, y_train)
        y = linreg.predict(poly.fit_transform(np.linspace(0,10,100).reshape(100,1)))
        result[i, :] = y
    return result

def answer_two():

    r2_train = np.zeros(10)
    r2_test = np.zeros(10)
    for degree in range(10):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train.reshape(11,1))
        linreg = LinearRegression().fit(X_poly, y_train)
        r2_train[degree] = linreg.score(X_poly, y_train)
        X_test_poly = poly.fit_transform(X_test.reshape(4,1))
        r2_test[degree] = linreg.score(X_test_poly, y_test)
    return (r2_train, r2_test)

def answer_three():
    
    r2_train, r2_test = answer_two()
    degrees = np.arange(0, 10)
    plt.figure()
    plt.plot(degrees, r2_train, degrees, r2_test)

def answer_four():
    
    poly = PolynomialFeatures(degree=12)
    X_train_poly = poly.fit_transform(X_train.reshape(11,1)) 
    X_test_poly = poly.fit_transform(X_test.reshape(4,1))
    
    linreg = LinearRegression().fit(X_train_poly, y_train)
    LinearRegression_R2_test_score = linreg.score(X_test_poly, y_test)
    
    linlasso = Lasso(alpha=0.01, max_iter = 10000).fit(X_train_poly, y_train)
    Lasso_R2_test_score = linlasso.score(X_test_poly, y_test)    
    
    return (LinearRegression_R2_test_score, Lasso_R2_test_score)

def answer_five():
    
    mush_df = pd.read_csv('mushrooms.csv')
    mush_df2 = pd.get_dummies(mush_df)

    X_mush = mush_df2.iloc[:,2:]
    y_mush = mush_df2.iloc[:,1]    
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)
    
    clf = DecisionTreeClassifier().fit(X_train2, y_train2)
    
    features = []
    for feature, importance in zip(X_train2.columns, clf.feature_importances_):
        features.append((importance, feature))
    features.sort(reverse=True)
    
    return [feature[1] for feature in features[:5]]

def answer_six():
    
    svc = SVC(random_state=0)
    gamma = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(svc,X_subset,y_subset, param_name='gamma',param_range=gamma,scoring='accuracy')
    train_scores = train_scores.mean(axis=1)
    test_scores = test_scores.mean(axis=1)

    return train_scores, test_scores

def answer_seven():
    
    train_scores, test_scores = answer_six()
    gamma = np.logspace(-4,1,6)
    plt.figure()
    plt.plot(gamma, train_scores, 'b--', gamma, test_scores, 'g-')

if __name__ == '__main__':   
        
    np.random.seed(0)
    n = 15
    x = np.linspace(0,10,n) + np.random.randn(n)/5
    y = np.sin(x)+x/6 + np.random.randn(n)/10
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    
    result = answer_one()
    r2_train, r2_test = answer_two()
    answer_three()
    LinearRegression_R2_test_score, Lasso_R2_test_score = answer_four()
    
    mush_df = pd.read_csv('mushrooms.csv')
    mush_df2 = pd.get_dummies(mush_df)

    X_mush = mush_df2.iloc[:,2:]
    y_mush = mush_df2.iloc[:,1]
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)
    X_subset = X_test2
    y_subset = y_test2
    
    answer_five()
    train_scores, test_scores = answer_six()
    answer_seven()
    