from numpy import genfromtxt
from sklearn import ensemble
from sklearn import model_selection
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neural_network
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def build_arrays(data):
    '''
    Splits the array of parameters and classes into two different ones
    :param data: get's the data array which is extracted from the csv
    :return: x: parameters (eeg etc.) y: hypno values
    '''
    x = []
    y = []
    for i in range(0, len(data)):
        x.append(data_array[i][4:12])
        y.append(data_array[i][3])
        pass
    return x, y


def class_accuracy(cm):
    cm_len = len(cm)
    acc_array = []
    sum = 0
    for i in range(0, cm_len):
        for j in range(0, cm_len):
            sum = sum + cm[j][i]
        if sum != 0: acc = (cm[i][i]*100/ sum)
        else: acc = 0
        acc_array.append(acc)
        sum = 0
    return acc_array

def class_recall(cm):
    cm_len = len(cm)
    rec_array = []
    sum = 0
    for i in range(0, cm_len):
        for j in range(0, cm_len):
            sum = sum + cm[i][j]
        if sum != 0: acc = (cm[i][i]*100/ sum)
        else: acc = 0
        rec_array.append(acc)
        sum = 0
    return rec_array


# read data
data_array = genfromtxt('data/Features_SC.txt', delimiter=',')

# create array
x, y = build_arrays(data_array)

# split array to test and train sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=100, shuffle=True)

# Decision tree with gini
classi_gini = DecisionTreeClassifier(criterion="gini", random_state=100, min_samples_leaf=5)
classi_gini.fit(x_train, y_train)

# Decision tree with information gain
classi_gain = DecisionTreeClassifier(criterion="entropy", random_state=100, min_samples_leaf=5)
classi_gain.fit(x_train, y_train)

classi_svm = svm.SVC()
classi_svm.fit(x_train, y_train)


classi_gaussi = GaussianNB()
classi_gaussi.fit(x_train, y_train)


# Random Forrest (experimental)
classi_rf = ensemble.RandomForestClassifier(n_estimators=50, criterion="gini", random_state=100,
                                            min_samples_leaf=5, bootstrap=True)
classi_rf.fit(x_train, y_train)
#result_rf_kf = model_selection.cross_val_score(classi_rf, x, y, cv=kfold)

classi_mlp = neural_network.MLPClassifier()
classi_mlp.fit(x_train, y_train)

# [classi_gini, classi_gain, classi_rf, classi_mlp, regr_1, classi_svm]

# Ada boost (experimental)
classi_ada = ensemble.AdaBoostClassifier(base_estimator=classi_rf, n_estimators=50, random_state=100, algorithm='SAMME')
classi_ada.estimators_ = [classi_gini, classi_gain, classi_mlp, classi_svm, classi_gaussi]
classi_ada.n_classes_ = 6
classi_ada.fit(x_train, y_train)


# predict the test sets for each classifier
y_pred_gini = classi_gini.predict(x_test)
y_pred_gain = classi_gain.predict(x_test)
y_pred_svm = classi_svm.predict(x_test)
y_pred_gaussi = classi_gaussi.predict(x_test)
y_pred_ada = classi_ada.predict(x_test)
y_pred_rf = classi_rf.predict(x_test)


print('## Confusion Matrix DT Gini')
cmat_gini = confusion_matrix(y_test, y_pred_gini)
print cmat_gini
print "Gini Accuracy is ", accuracy_score(y_test, y_pred_gini) * 100
print "Precision: ", class_accuracy(cmat_gini)
print "Recall: ", class_recall(cmat_gini)

print('## Confusion Matrix DT Gain')
cmat_gain = confusion_matrix(y_test, y_pred_gain)
print cmat_gain
print "Gain Accuracy is ", accuracy_score(y_test, y_pred_gain) * 100
print "Precision: ", class_accuracy(cmat_gain)
print "Recall: ", class_recall(cmat_gain)

print('## Confusion Matrix SVM')
cmat_svm = confusion_matrix(y_test, y_pred_svm)
print cmat_svm
print "SVM Accuracy is ", accuracy_score(y_test, y_pred_svm) * 100
print "Precision: ", class_accuracy(cmat_svm)
print "Recall: ", class_recall(cmat_svm)

print('## Confusion Matrix Gaussi')
cmat_gaussi = confusion_matrix(y_test, y_pred_gain)
print cmat_gaussi
print "Gaussi Accuracy is ", accuracy_score(y_test, y_pred_gaussi) * 100
print "Precision: ", class_accuracy(cmat_gaussi)
print "Recall: ", class_recall(cmat_gaussi)

print('## Confusion Matrix Ada Boost')
cmat_ada = confusion_matrix(y_test, y_pred_ada)
print cmat_ada
print "ADA Accuracy is ", accuracy_score(y_test, y_pred_ada) * 100
print "Precision: ", class_accuracy(cmat_ada)
print "Recall: ", class_recall(cmat_ada)

print('## Confusion Matrix Random Forrest')
cmat_rf = confusion_matrix(y_test, y_pred_rf)
print cmat_rf
print "RF Accuracy is ", accuracy_score(y_test, y_pred_rf) * 100
print "Precision: ", class_accuracy(cmat_rf)
print "Recall: ", class_recall(cmat_rf)

