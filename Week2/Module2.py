import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.datasets import make_classification, make_blobs, load_breast_cancer, make_regression, make_friedman1, load_iris
from matplotlib.colors import ListedColormap
from adspy_shared_utilities import load_crime_dataset, plot_two_class_knn, plot_class_regions_for_classifier_subplot, plot_class_regions_for_classifier, plot_decision_tree, plot_feature_importances
from sklearn.tree import DecisionTreeClassifier

def knn():
    
    fruits = pd.read_table('fruit_data_with_colors.txt')
    np.set_printoptions(precision=2)
    fruits = pd.read_table('fruit_data_with_colors.txt')

    feature_names_fruits = ['height', 'width', 'mass', 'color_score']
    X_fruits = fruits[feature_names_fruits]
    y_fruits = fruits['fruit_label']
    target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']
    
    X_fruits_2d = fruits[['height', 'width']]
    y_fruits_2d = fruits['fruit_label']

    X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train_scaled, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}' .format(knn.score(X_train_scaled, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}' .format(knn.score(X_test_scaled, y_test)))

    example_fruit = [[5.5, 2.2, 10, 0.70]]
    example_fruit_scaled = scaler.transform(example_fruit)
    print ('Predicted fruit type for ', example_fruit, ' is ', target_names_fruits[knn.predict(example_fruit_scaled)[0]-1])


def regression_data():
    
    cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
    plt.figure(1)
    plt.title('Sample regression problem with one input variable')
    X_R1, y_R1 = make_regression(n_samples = 100, n_features=1, n_informative=1, bias = 150.0, noise = 30, random_state=0)
    plt.scatter(X_R1, y_R1, marker= 'o', s=50)
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)
    knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)
    print(knnreg.predict(X_test))
    print('R-squared test score: {:.3f}' .format(knnreg.score(X_test, y_test)))

    fig, subaxes = plt.subplots(1, 2, figsize=(8,4))
    X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state = 0)

    for thisaxis, K in zip(subaxes, [1, 3]):
        knnreg = KNeighborsRegressor(n_neighbors = K).fit(X_train, y_train)
        y_predict_output = knnreg.predict(X_predict_input)
        thisaxis.set_xlim([-2.5, 0.75])
        thisaxis.plot(X_predict_input, y_predict_output, '^', markersize = 10, label='Predicted', alpha=0.8)
        thisaxis.plot(X_train, y_train, 'o', label='True Value', alpha=0.8)
        thisaxis.set_xlabel('Input feature')
        thisaxis.set_ylabel('Target value')
        thisaxis.set_title('KNN regression (K={})'.format(K))
        thisaxis.legend()
    plt.tight_layout()
    plt.show()

    fig, subaxes = plt.subplots(5, 1, figsize=(5,20))
    X_predict_input = np.linspace(-3, 3, 500).reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)

    for thisaxis, K in zip(subaxes, [1, 3, 7, 15, 55]):
        knnreg = KNeighborsRegressor(n_neighbors = K).fit(X_train, y_train)
        y_predict_output = knnreg.predict(X_predict_input)
        train_score = knnreg.score(X_train, y_train)
        test_score = knnreg.score(X_test, y_test)
        thisaxis.plot(X_predict_input, y_predict_output)
        thisaxis.plot(X_train, y_train, 'o', alpha=0.9, label='Train')
        thisaxis.plot(X_test, y_test, '^', alpha=0.9, label='Test')
        thisaxis.set_xlabel('Input feature')
        thisaxis.set_ylabel('Target value')
        thisaxis.set_title('KNN Regression (K={}) Train $R^2 = {:.3f}$,  Test $R^2 = {:.3f}$' .format(K, train_score, test_score))
        thisaxis.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

def binary_data():

    plt.figure()
    plt.title('Sample binary classification problem with two informative features')
    X_C2, y_C2 = make_classification(n_samples = 100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, flip_y = 0.1, class_sep = 0.5, random_state=0)
    plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2, marker= 'o', s=50, cmap=cmap_bold)
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)
    plot_two_class_knn(X_train, y_train, 1, 'uniform', X_test, y_test)
    plot_two_class_knn(X_train, y_train, 3, 'uniform', X_test, y_test)
    plot_two_class_knn(X_train, y_train, 11, 'uniform', X_test, y_test)
    plt.show()

def non_linear_data():
    
    X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
    y_D2 = y_D2 % 2
    plt.figure()
    plt.title('Sample binary classification problem with non-linearly separable classes')
    plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker= 'o', s=50, cmap=cmap_bold)
    plt.show()
    return X_D2, y_D2

def linear():
    
    X_R1, y_R1 = make_regression(n_samples = 100, n_features=1, n_informative=1, bias = 150.0, noise = 30, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
    print('linear model coeff (w): {}' .format(linreg.coef_))
    print('linear model intercept (b): {:.3f}' .format(linreg.intercept_))
    print('R-squared score (training): {:.3f}' .format(linreg.score(X_train, y_train)))
    print('R-squared score (test): {:.3f}' .format(linreg.score(X_test, y_test)))

    plt.figure(figsize=(5,4))
    plt.scatter(X_R1, y_R1, marker= 'o', s=50, alpha=0.8)
    plt.plot(X_R1, linreg.coef_ * X_R1 + linreg.intercept_, 'r-')
    plt.title('Least-squares linear regression')
    plt.xlabel('Feature value (x)')
    plt.ylabel('Target value (y)')
    plt.show()
    
    X_crime, y_crime = load_crime_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
    print('Crime dataset') 
    print('linear model intercept: {}' .format(linreg.intercept_))
    print('linear model coeff:\n{}' .format(linreg.coef_))
    print('R-squared score (training): {:.3f}' .format(linreg.score(X_train, y_train)))
    print('R-squared score (test): {:.3f}' .format(linreg.score(X_test, y_test)))
    
def ridge():
    
    X_crime, y_crime = load_crime_dataset()
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
    print('Crime dataset')
    print('ridge regression linear model intercept: {}' .format(linridge.intercept_))
    print('ridge regression linear model coeff:\n{}' .format(linridge.coef_))
    print('R-squared score (training): {:.3f}' .format(linridge.score(X_train_scaled, y_train)))
    print('R-squared score (test): {:.3f}' .format(linridge.score(X_test_scaled, y_test)))
    print('Number of non-zero features: {}' .format(np.sum(linridge.coef_ != 0)))
    
    print('Ridge regression: effect of alpha regularization parameter\n')
    for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
        linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, y_train)
        r2_train = linridge.score(X_train_scaled, y_train)
        r2_test = linridge.score(X_test_scaled, y_test)
        num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
        print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, r-squared training: {:.2f}, r-squared test: {:.2f}' .format(this_alpha, num_coeff_bigger, r2_train, r2_test))
    
def lasso():
    
    X_crime, y_crime = load_crime_dataset()
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)
    print('Crime dataset')
    print('lasso regression linear model intercept: {}' .format(linlasso.intercept_))
    print('lasso regression linear model coeff:{}' .format(linlasso.coef_))
    print('Non-zero features: {}' .format(np.sum(linlasso.coef_ != 0)))
    print('R-squared score (training): {:.3f}' .format(linlasso.score(X_train_scaled, y_train)))
    print('R-squared score (test): {:.3f}' .format(linlasso.score(X_test_scaled, y_test)))
    print('Features with non-zero weight (sorted by absolute magnitude):')

    for e in sorted (list(zip(list(X_crime), linlasso.coef_)), key = lambda e: -abs(e[1])):
        if e[1] != 0:
            print('\t{}, {:.3f}'.format(e[0], e[1]))
            
    print('Lasso regression: effect of alpha regularization parameter on number of features kept in final model\n')
    for alpha in [0.5, 1, 2, 3, 5, 10, 20, 50]:
        linlasso = Lasso(alpha, max_iter = 10000).fit(X_train_scaled, y_train)
        r2_train = linlasso.score(X_train_scaled, y_train)
        r2_test = linlasso.score(X_test_scaled, y_test)
    
        print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, r-squared test: {:.2f}' .format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))
        
def poly():
    
    plt.figure()
    plt.title('Complex regression problem with one input variable')
    X_F1, y_F1 = make_friedman1(n_samples = 100, n_features = 7, random_state=0)
    plt.scatter(X_F1[:, 2], y_F1, marker= 'o', s=50)
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X_F1, y_F1, random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
    print('linear model coeff (w): {}' .format(linreg.coef_))
    print('linear model intercept (b): {:.3f}' .format(linreg.intercept_))
    print('R-squared score (training): {:.3f}' .format(linreg.score(X_train, y_train)))
    print('R-squared score (test): {:.3f}' .format(linreg.score(X_test, y_test)))

    print('\nNow we transform the original input data to add polynomial features up to degree 2 (quadratic)\n')
    poly = PolynomialFeatures(degree=2)
    X_F1_poly = poly.fit_transform(X_F1)
    X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1, random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
    print('(poly deg 2) linear model coeff (w):\n{}' .format(linreg.coef_))
    print('(poly deg 2) linear model intercept (b): {:.3f}' .format(linreg.intercept_))
    print('(poly deg 2) R-squared score (training): {:.3f}' .format(linreg.score(X_train, y_train)))
    print('(poly deg 2) R-squared score (test): {:.3f}\n' .format(linreg.score(X_test, y_test)))

    print('\nAddition of many polynomial features often leads to overfitting, so we often use polynomial features in combination with regression that has a regularization penalty, like ridge regression.\n')

    X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1, random_state = 0)
    linreg = Ridge().fit(X_train, y_train)
    print('(poly deg 2 + ridge) linear model coeff (w):\n{}' .format(linreg.coef_))
    print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}' .format(linreg.intercept_))
    print('(poly deg 2 + ridge) R-squared score (training): {:.3f}' .format(linreg.score(X_train, y_train)))
    print('(poly deg 2 + ridge) R-squared score (test): {:.3f}' .format(linreg.score(X_test, y_test)))

def logistic():
    
    fruits = pd.read_table('fruit_data_with_colors.txt')
    X_fruits_2d = fruits[['height', 'width']]
    y_fruits_2d = fruits['fruit_label']
    
    fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
    y_fruits_apple = y_fruits_2d == 1 
    X_train, X_test, y_train, y_test = (train_test_split(X_fruits_2d.as_matrix(), y_fruits_apple.as_matrix(), random_state = 0))
    clf = LogisticRegression(C=100).fit(X_train, y_train)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, 'Logistic regression for binary classification\nFruit dataset: Apple vs others', subaxes)
    
    h = 6
    w = 8
    print('A fruit with height {} and width {} is predicted to be: {}' .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))

    h = 10
    w = 7
    print('A fruit with height {} and width {} is predicted to be: {}' .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))
    subaxes.set_xlabel('height')
    subaxes.set_ylabel('width')

    print('Accuracy of Logistic regression classifier on training set: {:.2f}' .format(clf.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}' .format(clf.score(X_test, y_test)))
    
    X_C2, y_C2 = make_classification(n_samples = 100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, flip_y = 0.1, class_sep = 0.5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
    fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
    clf = LogisticRegression().fit(X_train, y_train)
    title = 'Logistic regression, simple synthetic dataset C = {:.3f}'.format(1.0)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)

    print('Accuracy of Logistic regression classifier on training set: {:.2f}' .format(clf.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}' .format(clf.score(X_test, y_test)))
    
    X_train, X_test, y_train, y_test = (train_test_split(X_fruits_2d.as_matrix(),  y_fruits_apple.as_matrix(), random_state=0))
    fig, subaxes = plt.subplots(3, 1, figsize=(4, 10))

    for this_C, subplot in zip([0.1, 1, 100], subaxes):
        clf = LogisticRegression(C=this_C).fit(X_train, y_train)
        title ='Logistic regression (apple vs rest), C = {:.3f}'.format(this_C)
    
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test, y_test, title, subplot)
    plt.tight_layout()
    
    cancer = load_breast_cancer()
    X_cancer, y_cancer = load_breast_cancer(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
    clf = LogisticRegression().fit(X_train, y_train)
    print('Breast cancer dataset')
    print('Accuracy of Logistic regression classifier on training set: {:.2f}' .format(clf.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}' .format(clf.score(X_test, y_test)))

def svm():
    
    X_C2, y_C2 = make_classification(n_samples = 100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, flip_y = 0.1, class_sep = 0.5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)

    fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
    this_C = 1.0
    clf = SVC(kernel = 'linear', C=this_C).fit(X_train, y_train)
    title = 'Linear SVC, C = {:.3f}'.format(this_C)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)

    X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
    fig, subaxes = plt.subplots(1, 2, figsize=(8, 4))

    for this_C, subplot in zip([0.00001, 100], subaxes):
        clf = LinearSVC(C=this_C).fit(X_train, y_train)
        title = 'Linear SVC, C = {:.5f}'.format(this_C)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subplot)
    plt.tight_layout()

    cancer = load_breast_cancer()
    X_cancer, y_cancer = load_breast_cancer(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
    clf = LinearSVC().fit(X_train, y_train)
    print('Breast cancer dataset')
    print('Accuracy of Linear SVC classifier on training set: {:.2f}' .format(clf.score(X_train, y_train)))
    print('Accuracy of Linear SVC classifier on test set: {:.2f}' .format(clf.score(X_test, y_test)))

def multiclass():
    
    fruits = pd.read_table('fruit_data_with_colors.txt')
    X_fruits_2d = fruits[['height', 'width']]
    y_fruits_2d = fruits['fruit_label']
    target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']
    X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d, y_fruits_2d, random_state = 0)
    clf = LinearSVC(C=5, random_state = 67).fit(X_train, y_train)
    print('Coefficients:\n', clf.coef_)
    print('Intercepts:\n', clf.intercept_)
    
    plt.figure(figsize=(6,6))
    colors = ['r', 'g', 'b', 'y']
    cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#FFFF00'])
    plt.scatter(X_fruits_2d[['height']], X_fruits_2d[['width']], c=y_fruits_2d, cmap=cmap_fruits, edgecolor = 'black', alpha=.7)

    x_0_range = np.linspace(-10, 15)
    for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
        plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)
    
    plt.legend(target_names_fruits)
    plt.xlabel('height')
    plt.ylabel('width')
    plt.xlim(-2, 12)
    plt.ylim(-2, 15)
    plt.show()

def kernal():
    
    X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)    
    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
    plot_class_regions_for_classifier(SVC().fit(X_train, y_train), X_train, y_train, None, None, 'Support Vector Classifier: RBF kernel')
    plot_class_regions_for_classifier(SVC(kernel = 'poly', degree = 3) .fit(X_train, y_train), X_train, y_train, None, None, 'Support Vector Classifier: Polynomial kernel, degree = 3')
    
    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
    fig, subaxes = plt.subplots(3, 1, figsize=(4, 11))

    for this_gamma, subplot in zip([0.01, 1.0, 10.0], subaxes):
        clf = SVC(kernel = 'rbf', gamma=this_gamma).fit(X_train, y_train)
        title = 'Support Vector Classifier: \nRBF kernel, gamma = {:.2f}'.format(this_gamma)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subplot)
        plt.tight_layout()
    
    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
    fig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=50)

    for this_gamma, this_axis in zip([0.01, 1, 5], subaxes):    
        for this_C, subplot in zip([0.1, 1, 15, 250], this_axis):
            title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
            clf = SVC(kernel = 'rbf', gamma = this_gamma, C = this_C).fit(X_train, y_train)
            plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test, y_test, title, subplot)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    cancer = load_breast_cancer()
    X_cancer, y_cancer = load_breast_cancer(return_X_y = True)        
    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
    clf = SVC(C=10).fit(X_train, y_train)
    
    print('Breast cancer dataset (unnormalized features)')
    print('Accuracy of RBF-kernel SVC on training set: {:.2f}' .format(clf.score(X_train, y_train)))
    print('Accuracy of RBF-kernel SVC on test set: {:.2f}' .format(clf.score(X_test, y_test)))
    
    cancer = load_breast_cancer()
    X_cancer, y_cancer = load_breast_cancer(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(C=10).fit(X_train_scaled, y_train)
    print('Breast cancer dataset (unnormalized features)')
    print('Accuracy of RBF-kernel SVC on training set: {:.2f}' .format(clf.score(X_train_scaled, y_train)))
    print('Accuracy of RBF-kernel SVC on test set: {:.2f}' .format(clf.score(X_test_scaled, y_test)))
    
def crossvalidation():
    
    fruits = pd.read_table('fruit_data_with_colors.txt')
    X_fruits_2d = fruits[['height', 'width']]
    y_fruits_2d = fruits['fruit_label']
    clf = KNeighborsClassifier(n_neighbors = 5)
    X = X_fruits_2d.as_matrix()
    y = y_fruits_2d.as_matrix()
    cv_scores = cross_val_score(clf, X, y)

    print('Cross-validation scores (3-fold):', cv_scores)
    print('Mean cross-validation score (3-fold): {:.3f}' .format(np.mean(cv_scores)))
    
    param_range = np.logspace(-3, 3, 4)
    train_scores, test_scores = validation_curve(SVC(), X, y, param_name='gamma', param_range=param_range, cv=3)
    print(train_scores)
    print(test_scores)
    
    plt.figure()
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title('Validation Curve with SVM')
    plt.xlabel('$\gamma$ (gamma)')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label='Training score', color='darkorange', lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=lw)
    plt.semilogx(param_range, test_scores_mean, label='Cross-validation score', color='navy', lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=lw)
    plt.legend(loc='best')
    plt.show()

def decisiontree():
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 3)
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    print('Accuracy of Decision Tree classifier on training set: {:.2f}' .format(clf.score(X_train, y_train)))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}' .format(clf.score(X_test, y_test)))

    clf2 = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)
    print('Accuracy of Decision Tree classifier on training set: {:.2f}' .format(clf2.score(X_train, y_train)))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}' .format(clf2.score(X_test, y_test)))

    plot_decision_tree(clf, iris.feature_names, iris.target_names)
    plot_decision_tree(clf2, iris.feature_names, iris.target_names)
    
    plt.figure(figsize=(10,4), dpi=80)
    plot_feature_importances(clf, iris.feature_names)
    plt.show()

    print('Feature importances: {}'.format(clf.feature_importances_))
    
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 0)
    fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))
    pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
    tree_max_depth = 4

    for pair, axis in zip(pair_list, subaxes):
        X = X_train[:, pair]
        y = y_train    
        clf = DecisionTreeClassifier(max_depth=tree_max_depth).fit(X, y)
        title = 'Decision Tree, max_depth = {:d}'.format(tree_max_depth)
        plot_class_regions_for_classifier_subplot(clf, X, y, None, None, title, axis, iris.target_names)
        axis.set_xlabel(iris.feature_names[pair[0]])
        axis.set_ylabel(iris.feature_names[pair[1]])
    
    plt.tight_layout()
    plt.show()
    
    cancer = load_breast_cancer()
    X_cancer, y_cancer = load_breast_cancer(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
    clf = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 8, random_state = 0).fit(X_train, y_train)
    plot_decision_tree(clf, cancer.feature_names, cancer.target_names)
    
    print('Breast cancer dataset: decision tree')
    print('Accuracy of DT classifier on training set: {:.2f}' .format(clf.score(X_train, y_train)))
    print('Accuracy of DT classifier on test set: {:.2f}' .format(clf.score(X_test, y_test)))

    plt.figure(figsize=(10,6),dpi=80)
    plot_feature_importances(clf, cancer.feature_names)
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':    
    
    knn()
    regression_data()
    linear()
    ridge()
    lasso()
    poly()
    logistic()
    svm()
    #multiclass()
    kernal()
    crossvalidation()
    decisiontree()