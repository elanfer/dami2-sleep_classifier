import numpy as np
from numpy import genfromtxt
from sklearn import ensemble
from sklearn import model_selection
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def class_accuracy(cm):
    cm_len = len(cm)
    acc_array = []
    sum = np.asarray(cm).sum(axis=0)
    for i in range(0, cm_len):
        acc = cm[i][i] * 100 / sum[i]
        acc_array.append(acc)
    return acc_array


def class_recall(cm):
    cm_len = len(cm)
    acc_array = []
    sum = np.asarray(cm).sum(axis=1)
    for i in range(0, cm_len):
        acc = cm[i][i] * 100 / sum[i]
        acc_array.append(acc)
    return acc_array



# read data
data_array = genfromtxt('../data/Features_SC.txt', delimiter=',')
end = np.asarray(data_array).shape[1]
y = np.asarray(data_array[:, 3])
x = np.asarray(data_array[:, 4:end])

# normalization:
'''
x = np.abs(x)
ten = np.zeros(x.shape[0]) + 10
x[:,0:5] = np.log10(x[:,0:5])
x[:,6:8] = np.log10(x[:,6:8])
x[x == -np.inf] = 0
x = np.abs(x)
x = normalize(x, norm='max', axis=0)
'''

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=100, shuffle=True)
kfold = model_selection.KFold(n_splits=10, shuffle=True)

label = [int(i) for i in y]

'''
# PCA
pca1 = PCA().fit(x)
plt.figure(1)
plt.clf()
plt.title('PCA Spectrum ')
plt.grid(True)
# plt.xscale('log')
plt.plot(pca1.explained_variance_ratio_, linewidth=2)
plt.xlabel('n Components')
plt.ylabel('Explained Variance')

x = PCA(n_components=4).fit_transform(x)

plt.show()
print(pca1.explained_variance_ratio_)
print(pca1.singular_values_)
'''


# Multi Layer Perceptron
classi_mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=None)
classi_mlp.fit(x_train, y_train)

# Linear Discriminant Analysis (LDA)
classi_lda = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd',
                                        store_covariance=False, tol=0.0001)
classi_lda.fit(x_train, y_train)

# Decision tree with gini
classi_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
classi_gini.fit(x_train, y_train)

# Decision tree with information gain
classi_gain = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=5, min_samples_leaf=5)
classi_gain.fit(x_train, y_train)

# Support Vector machine with non linear kernel
classi_svm = svm.SVC()
classi_svm.fit(x_train, y_train)

# Ada boost (experimental)
classi_ada = ensemble.AdaBoostClassifier(n_estimators=50, random_state=100)
classi_ada.fit(x_train, y_train)

# Random Forrest (experimental)
classi_rf = ensemble.RandomForestClassifier(n_estimators=50, criterion="gini", random_state=100, max_depth=5,
                                            min_samples_leaf=5, bootstrap=True)
classi_rf.fit(x_train, y_train)


# predict the test sets for each classifier
y_pred_mlp = classi_mlp.predict(x_test)
y_pred_lda = classi_lda.predict(x_test)
y_pred_gini = classi_gini.predict(x_test)
y_pred_gain = classi_gain.predict(x_test)
y_pred_svm = classi_svm.predict(x_test)
y_pred_ada = classi_ada.predict(x_test)
y_pred_rf = classi_ada.predict(x_test)


print('## Confusion Matrix DT Gini')
cmat_gini = confusion_matrix(y_test, y_pred_gini)
print (cmat_gini)
print ("Gini Accuracy is ", accuracy_score(y_test, y_pred_gini) * 100)
print ("Precision: ", class_accuracy(cmat_gini))
print ("Recall: ", class_accuracy(cmat_gini))
print('## Confusion Matrix DT Gain')
cmat_gain = confusion_matrix(y_test, y_pred_gain)
print (cmat_gain)
print ("Gain Accuracy is ", accuracy_score(y_test, y_pred_gain) * 100)
print ("Precision: ", class_accuracy(cmat_gain))
print ("Recall: ", class_accuracy(cmat_gain))
print('## Confusion Matrix SVM')
cmat_svm = confusion_matrix(y_test, y_pred_svm)
print (cmat_svm)
print ("SVM Accuracy is ", accuracy_score(y_test, y_pred_svm) * 100)
print ("Precision: ", class_accuracy(cmat_svm))
print ("Recall: ", class_accuracy(cmat_svm))
print('## Confusion Matrix Ada Boost')
cmat_ada = confusion_matrix(y_test, y_pred_ada)
print (cmat_ada)
print ("ADA Accuracy is ", accuracy_score(y_test, y_pred_ada) * 100)
print ("Precision: ", class_accuracy(cmat_ada))
print ("Recall: ", class_accuracy(cmat_ada))
print('## Confusion Matrix Random Forrest')
cmat_rf = confusion_matrix(y_test, y_pred_rf)
print (cmat_rf)
print ("RF Accuracy is ", accuracy_score(y_test, y_pred_rf) * 100)
print ("Precision: ", class_accuracy(cmat_rf))
print ("Recall: ", class_accuracy(cmat_rf))
