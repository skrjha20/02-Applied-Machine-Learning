import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc, mean_squared_error
from sklearn.metrics import r2_score, roc_auc_score, scorer
from sklearn.svm import SVC

def answer_one():
    
    data = pd.read_csv('fraud_data.csv')
    count = data['Class'].value_counts()
    count = count.rename(index={0: 'not_fraud', 1: 'fraud'})
    ratio = np.float(count['fraud']/count['not_fraud'])
    return ratio
    
def answer_two():
    
    dummy = DummyClassifier().fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return accuracy, recall
    
def answer_three():
    
    svm = SVC().fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)   
    return accuracy, recall, precision

def answer_four():
    
    svm = SVC(C = 1e9, gamma = 1e-07).fit(X_train, y_train)
    y_pred = svm.decision_function(X_test) > -220
    confusion = confusion_matrix(y_test, y_pred)
    return confusion
    
def answer_five():
    
    lr = LogisticRegression().fit(X_train, y_train)
    lr_predicted = lr.predict(X_test)
    print(lr.score(X_test, y_test))
    print(confusion_matrix(y_test, lr_predicted))
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_predicted)))
    print('Precision: {:.2f}'.format(precision_score(y_test, lr_predicted)))
    print('Recall: {:.2f}'.format(recall_score(y_test, lr_predicted)))
    print('F1: {:.2f}'.format(f1_score(y_test, lr_predicted)))
    print(classification_report(y_test, lr_predicted, target_names=['not 1', '1']))
    
    y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)
    y_score_list = list(zip(y_test, y_scores_lr))
    #print y_score_list
    y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
    y_proba_list = list(zip(y_test[0:20], y_proba_lr[0:20,1]))
    #print y_proba_list
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]
    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.axes().set_aspect('equal')
    plt.show()
    
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    
if __name__ == '__main__':    
    
    #ratio = answer_one()
    
    data = pd.read_csv('fraud_data.csv')
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
    
    #accuracy, recall = answer_two()
    accuracy, recall, precision = answer_three()
    confusion = answer_four()
    answer_five()