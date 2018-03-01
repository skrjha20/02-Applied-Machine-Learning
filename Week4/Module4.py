import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs, make_regression, make_friedman1
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import load_crime_dataset, plot_class_regions_for_classifier, plot_class_regions_for_classifier_subplot, plot_labelled_scatter
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

def naive_bayes():
    
    cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
    fruits = pd.read_table('fruit_data_with_colors.txt')
    feature_names_fruits = ['height', 'width', 'mass', 'color_score']
    X_fruits = fruits[feature_names_fruits]
    y_fruits = fruits['fruit_label']
    target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']
    
    X_fruits_2d = fruits[['height', 'width']]
    y_fruits_2d = fruits['fruit_label']
    
    plt.figure()
    plt.title('Sample regression problem with one input variable')
    X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,n_informative=1, bias = 150.0, noise = 30, random_state=0)
    plt.scatter(X_R1, y_R1, marker= 'o', s=50)
    plt.show()

    plt.figure()
    plt.title('Complex regression problem with one input variable')
    X_F1, y_F1 = make_friedman1(n_samples = 100, n_features = 7, random_state=0)

    plt.scatter(X_F1[:, 2], y_F1, marker= 'o', s=50)
    plt.show()

    plt.figure()
    plt.title('Sample binary classification problem with two informative features')
    X_C2, y_C2 = make_classification(n_samples = 100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, flip_y = 0.1, class_sep = 0.5, random_state=0)
    plt.scatter(X_C2[:, 0], X_C2[:, 1], marker= 'o', c=y_C2, s=50, cmap=cmap_bold)
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)
    nbclf = GaussianNB().fit(X_train, y_train)
    plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test, 'Gaussian Naive Bayes classifier: Dataset 1')

    # more difficult synthetic dataset for classification (binary)
    # with classes that are not linearly separable
    X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
    y_D2 = y_D2 % 2
    plt.figure()
    plt.title('Sample binary classification problem with non-linearly separable classes')
    plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker= 'o', s=50, cmap=cmap_bold)
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)
    nbclf = GaussianNB().fit(X_train, y_train)
    plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test, 'Gaussian Naive Bayes classifier: Dataset 2')

    # Breast cancer dataset for classification
    cancer = load_breast_cancer()
    (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
    nbclf = GaussianNB().fit(X_train, y_train)
    print('Breast cancer dataset')
    print('Accuracy of GaussianNB classifier on training set: {:.2f}' .format(nbclf.score(X_train, y_train)))

    print('Accuracy of GaussianNB classifier on test set: {:.2f}' .format(nbclf.score(X_test, y_test)))
    # Communities and Crime dataset
    (X_crime, y_crime) = load_crime_dataset()
    print('Crime dataset')
    print('Accuracy of GaussianNB classifier on training set: {:.2f}' .format(nbclf.score(X_train, y_train)))
    print('Accuracy of GaussianNB classifier on test set: {:.2f}' .format(nbclf.score(X_test, y_test)))

def random_forest():
    
    X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
    y_D2 = y_D2 % 2
    
    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)
    fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))
    clf = RandomForestClassifier().fit(X_train, y_train)
    title = 'Random Forest Classifier, complex binary dataset, default settings'
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test, y_test, title, subaxes)
    plt.show()
    
    fruits = pd.read_table('fruit_data_with_colors.txt')
    feature_names_fruits = ['height', 'width', 'mass', 'color_score']
    X_fruits = fruits[feature_names_fruits]
    y_fruits = fruits['fruit_label']
    target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']
    
    X_train, X_test, y_train, y_test = train_test_split(X_fruits.as_matrix(), y_fruits.as_matrix(), random_state = 0)
    fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

    title = 'Random Forest, fruits dataset, default settings'
    pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]

    for pair, axis in zip(pair_list, subaxes):
        X = X_train[:, pair]
        y = y_train    
        clf = RandomForestClassifier().fit(X, y)
        plot_class_regions_for_classifier_subplot(clf, X, y, None, None, title, axis, target_names_fruits)
        axis.set_xlabel(feature_names_fruits[pair[0]])
        axis.set_ylabel(feature_names_fruits[pair[1]])
    
    plt.tight_layout()
    plt.show()

    clf = RandomForestClassifier(n_estimators = 10, random_state=0).fit(X_train, y_train)
    print('Random Forest, Fruit dataset, default settings')
    print('Accuracy of RF classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of RF classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    
    cancer = load_breast_cancer()
    (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
        
    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
    clf = RandomForestClassifier(max_features = 8, random_state = 0)
    clf.fit(X_train, y_train)

    print('Breast cancer dataset')
    print('Accuracy of RF classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of RF classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

def gradient_boosting():
    
    X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
    y_D2 = y_D2 % 2
    
    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
    fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))

    clf = GradientBoostingClassifier().fit(X_train, y_train)
    title = 'GBDT, complex binary dataset, default settings'
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test, y_test, title, subaxes)
    plt.show()

    fruits = pd.read_table('fruit_data_with_colors.txt')
    feature_names_fruits = ['height', 'width', 'mass', 'color_score']
    X_fruits = fruits[feature_names_fruits]
    y_fruits = fruits['fruit_label']
    target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']
    
    X_train, X_test, y_train, y_test = train_test_split(X_fruits.as_matrix(), y_fruits.as_matrix(), random_state = 0)
    fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))
    
    pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
    for pair, axis in zip(pair_list, subaxes):
        X = X_train[:, pair]
        y = y_train
        clf = GradientBoostingClassifier().fit(X, y)
        plot_class_regions_for_classifier_subplot(clf, X, y, None, None, title, axis, target_names_fruits)
        axis.set_xlabel(feature_names_fruits[pair[0]])
        axis.set_ylabel(feature_names_fruits[pair[1]])
    
    plt.tight_layout()
    plt.show()
    clf = GradientBoostingClassifier().fit(X_train, y_train)

    print('GBDT, Fruit dataset, default settings')
    print('Accuracy of GBDT classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of GBDT classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

    cancer = load_breast_cancer()
    (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)    
    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

    clf = GradientBoostingClassifier(random_state = 0)
    clf.fit(X_train, y_train)
    print('Breast cancer dataset (learning_rate=0.1, max_depth=3)')
    print('Accuracy of GBDT classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of GBDT classifier on test set: {:.2f}\n'.format(clf.score(X_test, y_test)))

    clf = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 2, random_state = 0)
    clf.fit(X_train, y_train)

    print('Breast cancer dataset (learning_rate=0.01, max_depth=2)')
    print('Accuracy of GBDT classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of GBDT classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

def neural_network():
    
    #Synthetic dataset 1: single hidden layer
    X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
    y_D2 = y_D2 % 2

    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)
    fig, subaxes = plt.subplots(3, 1, figsize=(6,18))
    for units, axis in zip([1, 10, 100], subaxes):
        nnclf = MLPClassifier(hidden_layer_sizes = [units], solver='lbfgs', random_state = 0).fit(X_train, y_train)    
        title = 'Dataset 1: Neural net classifier, 1 layer, {} units'.format(units)
        plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train, X_test, y_test, title, axis)
        plt.tight_layout()
    
    # Synthetic dataset 1: two hidden layers
    nnclf = MLPClassifier(hidden_layer_sizes = [10, 10], solver='lbfgs', random_state = 0).fit(X_train, y_train)
    plot_class_regions_for_classifier(nnclf, X_train, y_train, X_test, y_test, 'Dataset 1: Neural net classifier, 2 layers, 10/10 units')
    
    #Regularization parameter: alpha
    fig, subaxes = plt.subplots(4, 1, figsize=(6, 23))
    for this_alpha, axis in zip([0.01, 0.1, 1.0, 5.0], subaxes):
        nnclf = MLPClassifier(solver='lbfgs', activation = 'tanh', alpha = this_alpha, hidden_layer_sizes = [100, 100], random_state = 0).fit(X_train, y_train)
        title = 'Dataset 2: NN classifier, alpha = {:.3f} '.format(this_alpha)
        plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train, X_test, y_test, title, axis)
        plt.tight_layout()
    
    #The effect of different choices of activation function    
    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)
    fig, subaxes = plt.subplots(3, 1, figsize=(6,18))
    for this_activation, axis in zip(['logistic', 'tanh', 'relu'], subaxes):
        nnclf = MLPClassifier(solver='lbfgs', activation = this_activation, alpha = 0.1, hidden_layer_sizes = [10, 10], random_state = 0).fit(X_train, y_train)
        title = 'Dataset 2: NN classifier, 2 layers 10/10, {} activation function'.format(this_activation)
        plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train, X_test, y_test, title, axis)
        plt.tight_layout()
    
    #Neural networks: Regression
    plt.figure()
    plt.title('Sample regression problem with one input variable')
    X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,n_informative=1, bias = 150.0, noise = 30, random_state=0)
    plt.scatter(X_R1, y_R1, marker= 'o', s=50)
    plt.show()
    
    fig, subaxes = plt.subplots(2, 3, figsize=(11,8), dpi=70)
    X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state = 0)

    for thisaxisrow, thisactivation in zip(subaxes, ['tanh', 'relu']):
        for thisalpha, thisaxis in zip([0.0001, 1.0, 100], thisaxisrow):
            mlpreg = MLPRegressor(hidden_layer_sizes = [100,100], activation = thisactivation, alpha = thisalpha, solver = 'lbfgs').fit(X_train, y_train)
            y_predict_output = mlpreg.predict(X_predict_input)
            thisaxis.set_xlim([-2.5, 0.75])
            thisaxis.plot(X_predict_input, y_predict_output, '^', markersize = 10)
            thisaxis.plot(X_train, y_train, 'o')
            thisaxis.set_xlabel('Input feature')
            thisaxis.set_ylabel('Target value')
            thisaxis.set_title('MLP regression\nalpha={}, activation={})' .format(thisalpha, thisactivation))
            plt.tight_layout()
    
    #Application to real-world dataset for classification
    cancer = load_breast_cancer()
    (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)  
    scaler = MinMaxScaler()

    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0, random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)
    print('Breast cancer dataset')
    print('Accuracy of NN classifier on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))
    print('Accuracy of NN classifier on test set: {:.2f}'.format(clf.score(X_test_scaled, y_test)))

def pca():
    
    cancer = load_breast_cancer()
    (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
    X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)  
    pca = PCA(n_components = 2).fit(X_normalized)
    X_pca = pca.transform(X_normalized)
    print(X_cancer.shape, X_pca.shape)

    plot_labelled_scatter(X_pca, y_cancer, ['malignant', 'benign'])
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.title('Breast Cancer Dataset PCA (n_components = 2)');

    fig = plt.figure(figsize=(8, 4))
    plt.imshow(pca.components_, interpolation = 'none', cmap = 'plasma')
    feature_names = list(cancer.feature_names)
    plt.gca().set_xticks(np.arange(-.5, len(feature_names)))
    plt.gca().set_yticks(np.arange(0.5, 2))
    plt.gca().set_xticklabels(feature_names, rotation=90, ha='left', fontsize=12)
    plt.gca().set_yticklabels(['First PC', 'Second PC'], va='bottom', fontsize=12)
    plt.colorbar(orientation='horizontal', ticks=[pca.components_.min(), 0, pca.components_.max()], pad=0.65)
    
    fruits = pd.read_table('fruit_data_with_colors.txt')
    feature_names_fruits = ['height', 'width', 'mass', 'color_score']
    X_fruits = fruits[feature_names_fruits]
    y_fruits = fruits['fruit_label']
    target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']
    
    X_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)  
    pca = PCA(n_components = 2).fit(X_normalized)
    X_pca = pca.transform(X_normalized) 
    plot_labelled_scatter(X_pca, y_fruits, ['apple','mandarin','orange','lemon'])
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.title('Fruits Dataset PCA (n_components = 2)');
    
def mds():
    
    fruits = pd.read_table('fruit_data_with_colors.txt')
    feature_names_fruits = ['height', 'width', 'mass', 'color_score']
    X_fruits = fruits[feature_names_fruits]
    y_fruits = fruits['fruit_label']
    target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']
    
    X_fruits_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)  
    mds = MDS(n_components = 2)
    X_fruits_mds = mds.fit_transform(X_fruits_normalized)

    plot_labelled_scatter(X_fruits_mds, y_fruits, ['apple', 'mandarin', 'orange', 'lemon'])
    plt.xlabel('First MDS feature')
    plt.ylabel('Second MDS feature')
    plt.title('Fruit sample dataset MDS');
    
    cancer = load_breast_cancer()
    (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
    X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)  
    mds = MDS(n_components = 2)
    X_mds = mds.fit_transform(X_normalized)

    plot_labelled_scatter(X_mds, y_cancer, ['malignant', 'benign'])
    plt.xlabel('First MDS dimension')
    plt.ylabel('Second MDS dimension')
    plt.title('Breast Cancer Dataset MDS (n_components = 2)');

def tsne():
    
    fruits = pd.read_table('fruit_data_with_colors.txt')
    feature_names_fruits = ['height', 'width', 'mass', 'color_score']
    X_fruits = fruits[feature_names_fruits]
    y_fruits = fruits['fruit_label']
    target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']
    X_fruits_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)  
    
    tsne = TSNE(random_state = 0)
    X_tsne = tsne.fit_transform(X_fruits_normalized)
    plot_labelled_scatter(X_tsne, y_fruits, ['apple', 'mandarin', 'orange', 'lemon'])
    plt.xlabel('First t-SNE feature')
    plt.ylabel('Second t-SNE feature')
    plt.title('Fruits dataset t-SNE');
    
    cancer = load_breast_cancer()
    (X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
    X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)  
    tsne = TSNE(random_state = 0)
    X_tsne = tsne.fit_transform(X_normalized)

    plot_labelled_scatter(X_tsne, y_cancer, ['malignant', 'benign'])
    plt.xlabel('First t-SNE feature')
    plt.ylabel('Second t-SNE feature')
    plt.title('Breast cancer dataset t-SNE');

def kmeans_clustering():
    
    X, y = make_blobs(random_state = 10)
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    plot_labelled_scatter(X, kmeans.labels_, ['Cluster 1', 'Cluster 2', 'Cluster 3'])

    fruits = pd.read_table('fruit_data_with_colors.txt')
    X_fruits = fruits[['mass','width','height', 'color_score']].as_matrix()
    y_fruits = fruits[['fruit_label']] - 1    
    X_fruits_normalized = MinMaxScaler().fit(X_fruits).transform(X_fruits)  
    kmeans = KMeans(n_clusters = 4, random_state = 0)
    kmeans.fit(X_fruits_normalized)
    plot_labelled_scatter(X_fruits_normalized, kmeans.labels_, ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])

def agglomerative_clustering():
    
    X, y = make_blobs(random_state = 10)
    cls = AgglomerativeClustering(n_clusters = 3)
    cls_assignment = cls.fit_predict(X)
    plot_labelled_scatter(X, cls_assignment, ['Cluster 1', 'Cluster 2', 'Cluster 3'])

def dendogram():
    
    X, y = make_blobs(random_state = 10, n_samples = 10)
    plot_labelled_scatter(X, y, ['Cluster 1', 'Cluster 2', 'Cluster 3'])
    print(X)

    plt.figure()
    dendrogram(ward(X))
    plt.show()
    
def dbscan():
    
    X, y = make_blobs(random_state = 9, n_samples = 25)
    dbscan = DBSCAN(eps = 2, min_samples = 2)
    cls = dbscan.fit_predict(X)
    print("Cluster membership values:\n{}".format(cls))
    plot_labelled_scatter(X, cls + 1, ['Noise', 'Cluster 0', 'Cluster 1', 'Cluster 2'])

if __name__ == '__main__':    
    
    naive_bayes()
    random_forest()
    gradient_boosting()
    neural_network()
    pca()
    mds()
    tsne()
    kmeans_clustering()
    agglomerative_clustering()
    dendogram()
    dbscan()