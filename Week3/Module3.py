import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot, plot_class_regions_for_classifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc, mean_squared_error
from sklearn.metrics import r2_score, roc_auc_score, scorer
from sklearn.datasets import load_digits, load_diabetes
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.svm import SVC

from sklearn.metrics.scorer import SCORERS
def balanced_data():
    
    dataset = load_digits()
    X, y = dataset.data, dataset.target

    for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)):
        print(class_name,class_count)

def imbalanced_data():
    
    dataset = load_digits()
    X, y = dataset.data, dataset.target
    y_binary_imbalanced = y.copy()
    y_binary_imbalanced[y_binary_imbalanced != 1] = 0

    print('Original labels:\t', y[1:30])
    print('New binary labels:\t', y_binary_imbalanced[1:30])
    print(np.bincount(y_binary_imbalanced))
        
def svc_rbf_kernal():
    
    dataset = load_digits()
    X, y = dataset.data, dataset.target
    y_binary_imbalanced = y.copy()
    y_binary_imbalanced[y_binary_imbalanced != 1] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
    
    print('\nSVC RBF Kernal')
    svm = SVC(kernel='rbf', C=1).fit(X_train, y_train)
    svm_predicted = svm.predict(X_test) 
    print(svm.score(X_test, y_test))
    print(confusion_matrix(y_test, svm_predicted))
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, svm_predicted)))
    print('Precision: {:.2f}'.format(precision_score(y_test, svm_predicted)))
    print('Recall: {:.2f}'.format(recall_score(y_test, svm_predicted)))
    print('F1: {:.2f}'.format(f1_score(y_test, svm_predicted)))
    print(classification_report(y_test, svm_predicted, target_names=['not 1', '1']))
    y_scores_svm = svm.fit(X_train, y_train).decision_function(X_test)
    y_score_list = list(zip(y_test, y_scores_svm))
    print(y_score_list)
    
def dummy_most_frequent():
    
    dataset = load_digits()
    X, y = dataset.data, dataset.target
    y_binary_imbalanced = y.copy()
    y_binary_imbalanced[y_binary_imbalanced != 1] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
    
    print('\nDummy Most Frequent')
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    y_majority_predicted = dummy_majority.predict(X_test)
    print(dummy_majority.score(X_test, y_test))
    print(confusion_matrix(y_test, y_majority_predicted))
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_majority_predicted)))
    print('Precision: {:.2f}'.format(precision_score(y_test, y_majority_predicted)))
    print('Recall: {:.2f}'.format(recall_score(y_test, y_majority_predicted)))
    print('F1: {:.2f}'.format(f1_score(y_test, y_majority_predicted)))
    print(classification_report(y_test, y_majority_predicted, target_names=['not 1', '1']))

def dummy_stratified():
    
    dataset = load_digits()
    X, y = dataset.data, dataset.target
    y_binary_imbalanced = y.copy()
    y_binary_imbalanced[y_binary_imbalanced != 1] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
    
    print('\nDummy Stratified')
    dummy_classprop = DummyClassifier(strategy='stratified').fit(X_train, y_train)
    y_classprop_predicted = dummy_classprop.predict(X_test) 
    print(dummy_classprop.score(X_test, y_test))
    print(confusion_matrix(y_test, y_classprop_predicted))
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_classprop_predicted)))
    print('Precision: {:.2f}'.format(precision_score(y_test, y_classprop_predicted)))
    print('Recall: {:.2f}'.format(recall_score(y_test, y_classprop_predicted)))
    print('F1: {:.2f}'.format(f1_score(y_test, y_classprop_predicted)))
    print(classification_report(y_test, y_classprop_predicted, target_names=['not 1', '1']))
    
def svc_linear():
    
    dataset = load_digits()
    X, y = dataset.data, dataset.target
    y_binary_imbalanced = y.copy()
    y_binary_imbalanced[y_binary_imbalanced != 1] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
    
    print('\nSVC Linear')
    svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
    svm_predicted = svm.predict(X_test) 
    print(svm.score(X_test, y_test))
    print(confusion_matrix(y_test, svm_predicted))
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, svm_predicted)))
    print('Precision: {:.2f}'.format(precision_score(y_test, svm_predicted)))
    print('Recall: {:.2f}'.format(recall_score(y_test, svm_predicted)))
    print('F1: {:.2f}'.format(f1_score(y_test, svm_predicted)))
    print(classification_report(y_test, svm_predicted, target_names=['not 1', '1']))
    y_scores_svm = svm.fit(X_train, y_train).decision_function(X_test)
    y_score_list = list(zip(y_test, y_scores_svm))
    #print y_score_list
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_svm)
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
    
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_svm)
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

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    for g in [0.01, 0.1, 0.20, 1]:
        svm = SVC(gamma=g).fit(X_train, y_train)
        y_score_svm = svm.decision_function(X_test)
        fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
        roc_auc_svm = auc(fpr_svm, tpr_svm)
        accuracy_svm = svm.score(X_test, y_test)
        print("gamma = {:.2f}  accuracy = {:.2f}   AUC = {:.2f}".format(g, accuracy_svm, roc_auc_svm))
        plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7, label='SVM (gamma = {:0.2f}, area = {:0.2f})'.format(g, roc_auc_svm))

    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
    plt.legend(loc="lower right", fontsize=11)
    plt.title('ROC curve: (1-of-10 digits classifier)', fontsize=16)
    plt.axes().set_aspect('equal')    
    plt.show()
    
    
def logistic():
    
    dataset = load_digits()
    X, y = dataset.data, dataset.target
    y_binary_imbalanced = y.copy()
    y_binary_imbalanced[y_binary_imbalanced != 1] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
    
    print('\nLogistic')
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

def decision_tree():
    
    dataset = load_digits()
    X, y = dataset.data, dataset.target
    y_binary_imbalanced = y.copy()
    y_binary_imbalanced[y_binary_imbalanced != 1] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
    
    print('\nDecision Tree')
    dt = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
    tree_predicted = dt.predict(X_test)
    print(dt.score(X_test, y_test))
    print(confusion_matrix(y_test, tree_predicted))
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
    print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
    print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
    print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))
    print(classification_report(y_test, tree_predicted, target_names=['not 1', '1']))
    
    print('Micro-averaged precision = {:.2f} (treat instances equally)' .format(precision_score(y_test, tree_predicted, average = 'micro')))
    print('Macro-averaged precision = {:.2f} (treat classes equally)' .format(precision_score(y_test, tree_predicted, average = 'macro')))

    y_proba_dt = dt.fit(X_train, y_train).predict_proba(X_test)
    y_proba_list = list(zip(y_test, y_proba_dt))
    
def multiclass():
    
    dataset = load_digits()
    X,  y = dataset.data, dataset.target
    X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(X, y, random_state=0)

    svm = SVC(kernel = 'linear').fit(X_train_mc, y_train_mc)
    svm_predicted_mc = svm.predict(X_test_mc)
    confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)
    df_cm = pd.DataFrame(confusion_mc, index = [i for i in range(0,10)], columns = [i for i in range(0,10)])
    print(df_cm)
    
    plt.figure(figsize=(5.5,4))
    sns.heatmap(df_cm, annot=True)
    plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_mc, svm_predicted_mc)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    print(classification_report(y_test_mc, svm_predicted_mc))
    print('Micro-averaged precision = {:.2f} (treat instances equally)' .format(precision_score(y_test_mc, svm_predicted_mc, average = 'micro')))
    print('Macro-averaged precision = {:.2f} (treat classes equally)' .format(precision_score(y_test_mc, svm_predicted_mc, average = 'macro')))

    svm = SVC(kernel = 'rbf').fit(X_train_mc, y_train_mc)
    svm_predicted_mc = svm.predict(X_test_mc)
    confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)
    df_cm = pd.DataFrame(confusion_mc, index = [i for i in range(0,10)], columns = [i for i in range(0,10)])

    plt.figure(figsize = (5.5,4))
    sns.heatmap(df_cm, annot=True)
    plt.title('SVM RBF Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_mc, svm_predicted_mc)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    print(classification_report(y_test_mc, svm_predicted_mc))
    print('Micro-averaged precision = {:.2f} (treat instances equally)' .format(precision_score(y_test_mc, svm_predicted_mc, average = 'micro')))
    print('Macro-averaged precision = {:.2f} (treat classes equally)' .format(precision_score(y_test_mc, svm_predicted_mc, average = 'macro')))

def regression_evaluation():
    
    diabetes = load_diabetes()
    X = diabetes.data[:, None, 6]
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lm = LinearRegression().fit(X_train, y_train)
    lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)

    y_predict = lm.predict(X_test)
    y_predict_dummy_mean = lm_dummy_mean.predict(X_test)

    print('Linear model, coefficients: ', lm.coef_)
    print("Mean squared error (dummy): {:.2f}".format(mean_squared_error(y_test, 
                                                                     y_predict_dummy_mean)))
    print("Mean squared error (linear model): {:.2f}".format(mean_squared_error(y_test, y_predict)))
    print("r2_score (dummy): {:.2f}".format(r2_score(y_test, y_predict_dummy_mean)))
    print("r2_score (linear model): {:.2f}".format(r2_score(y_test, y_predict)))

    # Plot outputs
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_predict, color='green', linewidth=2)
    plt.plot(X_test, y_predict_dummy_mean, color='red', linestyle = 'dashed', linewidth=2, label = 'dummy')
    plt.show()  

def model_selection():
    
    dataset = load_digits()
    X, y = dataset.data, dataset.target == 1
    clf = SVC(kernel='linear', C=1)

    # accuracy is the default scoring metric
    print('Cross-validation (accuracy)', cross_val_score(clf, X, y, cv=5))
    # use AUC as scoring metric
    print('Cross-validation (AUC)', cross_val_score(clf, X, y, cv=5, scoring = 'roc_auc'))
    # use recall as scoring metric
    print('Cross-validation (recall)', cross_val_score(clf, X, y, cv=5, scoring = 'recall'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = SVC(kernel='rbf')
    grid_values = {'gamma': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}
    
    # default metric to optimize over grid parameters: accuracy
    grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)
    grid_clf_acc.fit(X_train, y_train)
    y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test) 
    
    print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
    print('Grid best score (accuracy): ', grid_clf_acc.best_score_)
    
    # alternative metric to optimize over grid parameters: AUC
    grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
    grid_clf_auc.fit(X_train, y_train)
    y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test) 
    
    print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
    print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
    print('Grid best score (AUC): ', grid_clf_auc.best_score_)
    print(sorted(list(scorer.SCORERS.keys())))

def two_feature_classification():
    
    dataset = load_digits()
    X, y = dataset.data, dataset.target == 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Create a two-feature input vector matching the example plot above
    # We jitter the points (add a small amount of random noise) in case there are areas
    # in feature space where many instances have the same features.
    jitter_delta = 0.25
    X_twovar_train = X_train[:,[20,59]]+ np.random.rand(X_train.shape[0], 2) - jitter_delta
    X_twovar_test  = X_test[:,[20,59]] + np.random.rand(X_test.shape[0], 2) - jitter_delta

    clf = SVC(kernel = 'linear').fit(X_twovar_train, y_train)
    grid_values = {'class_weight':['balanced', {1:2},{1:3},{1:4},{1:5},{1:10},{1:20},{1:50}]}
    plt.figure(figsize=(9,6))
    for i, eval_metric in enumerate(('precision','recall', 'f1','roc_auc')):
        grid_clf_custom = GridSearchCV(clf, param_grid=grid_values, scoring=eval_metric)
        grid_clf_custom.fit(X_twovar_train, y_train)
        print('Grid best parameter (max. {0}): {1}'.format(eval_metric, grid_clf_custom.best_params_))
        print('Grid best score ({0}): {1}'.format(eval_metric, grid_clf_custom.best_score_))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plot_class_regions_for_classifier_subplot(grid_clf_custom, X_twovar_test, y_test, None, None, None,  plt.subplot(2, 2, i+1))
        plt.title(eval_metric+'-oriented SVC')
        plt.tight_layout()
    plt.show()
    
    y_scores = clf.decision_function(X_twovar_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    plot_class_regions_for_classifier(clf, X_twovar_test, y_test)
    plt.title("SVC, class_weight = 'balanced', optimized for accuracy")
    plt.show()

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.title ("Precision-recall curve: SVC, class_weight = 'balanced'")
    plt.plot(precision, recall, label = 'Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize=12, fillstyle='none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.axes().set_aspect('equal')
    plt.show()
    print('At zero threshold, precision: {:.2f}, recall: {:.2f}'.format(closest_zero_p, closest_zero_r))
    
if __name__ == '__main__':    
    
    balanced_data()
    imbalanced_data()
    svc_rbf_kernal()
    dummy_most_frequent()
    dummy_stratified()
    svc_linear()
    logistic()
    decision_tree()
    multiclass()
    regression_evaluation()
    model_selection()
    two_feature_classification()
    
    