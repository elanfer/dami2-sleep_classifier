from sklearn.cross_validation import train_test_split
import csv

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def build_arrays(data):
    x = []
    y = []
    for i in range(0, len(data)):
        x.append(data_array[i][0:5])
        y.append(data_array[i][5])
        pass
    return x, y


# read data
data_array = []
with open('data/testrun.txt', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='\'')
    for row in spamreader:
        data_array.append(row)
x, y = build_arrays(data_array)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

classi_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
classi_gini.fit(x_train, y_train)
classi_gain = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
classi_gain.fit(x_train, y_train)

y_pred_gini = classi_gini.predict(x_test)
y_pred_gain = classi_gain.predict(x_test)

print "Gini Accuracy is ", accuracy_score(y_test, y_pred_gini) * 100
print "Gain Accuracy is ", accuracy_score(y_test, y_pred_gain) * 100
