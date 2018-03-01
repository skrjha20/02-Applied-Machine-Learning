import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def answer_zero():
    
    cancer = load_breast_cancer()
    return len(cancer['feature_names'])

def answer_one():
    
    X_cancer, y_cancer = load_breast_cancer(return_X_y = True)
    X_cancer = pd.DataFrame(X_cancer)
    y_cancer = pd.DataFrame(y_cancer)
    data = pd.concat([X_cancer, y_cancer],axis=1)
    data.columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                    'mean smoothness', 'mean compactness', 'mean concavity',
                    'mean concave points', 'mean symmetry', 'mean fractal dimension',
                    'radius error', 'texture error', 'perimeter error', 'area error',
                    'smoothness error', 'compactness error', 'concavity error',
                    'concave points error', 'symmetry error', 'fractal dimension error',
                    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                    'worst smoothness', 'worst compactness', 'worst concavity',
                    'worst concave points', 'worst symmetry', 'worst fractal dimension',
                    'target']
    return data 

def answer_two():
    
    cancerdf =  answer_one()
    count = cancerdf['target'].value_counts()
    count = count.rename(index={0: 'malignant', 1: 'benign'})
    return count

def answer_three():
    
    cancerdf = answer_one()
    X = cancerdf.loc[:, 'mean radius':'worst fractal dimension']
    y = cancerdf['target']
    return X, y

def answer_four():
    
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test

def answer_five():
    
    X_train, X_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors = 1)
    return knn.fit(X_train,y_train)

def answer_six():
    
    knn = answer_five()
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    cancer_prediction = knn.predict(means)
    return cancer_prediction

def answer_seven():
    
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    y_pred = knn.predict(X_test)
    return y_pred

def answer_eight():
    
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    score =  knn.score(X_test, y_test)
    accuracy = np.mean(score)
    return accuracy

if __name__ == '__main__':    
    
    length = answer_zero()
    cancerdf =  answer_one()
    class_count = answer_two()
    X, y = answer_three()
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    predict_mean = answer_six()
    y_pred = answer_seven()
    accuracy = answer_eight()
    