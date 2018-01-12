from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import t
from sklearn.preprocessing import normalize


def ttest(x1, x2, confidence=0.95):
    # return: statistic, p-value
    ind_t_test = stats.ttest_ind(x1, x2)
    N1 = len(x1)
    N2 = len(x2)
    # degrees of freedom
    df = N1 + N2 - 2
    std1 = x1.std()
    std2 = x2.std()
    std_N1N2 = sqrt(((N1 - 1) * (std1) ** 2 + (N2 - 1) * (std2) ** 2) / df)

    # mean dist
    w0 = abs(np.mean(x1) - np.mean(x2))
    MoE = t.ppf(confidence, df) * std_N1N2 * sqrt(1 / N1 + 1 / N2)

    print('The results of the independent t-test are: \n\tt-value = {:4.3f}\n\tp-value = {:4.8f}'.format(ind_t_test[0],
                                                                                                         ind_t_test[1]))
    print (
        '\nThe difference between groups is {:3.4f} [{:3.4f} to {:3.4f}] (mean [95% CI])'.format(w0, w0 - MoE,
                                                                                                 w0 + MoE))




# read data
data_array = np.genfromtxt('../data/Features_SC.txt', delimiter=',')
end = np.asarray(data_array).shape[1]
y = np.asarray(data_array[:, 3])
x = np.asarray(data_array[:, 4:end])

# normalization:
x = np.abs(x)
ten = np.zeros(x.shape[0]) + 10
x[:, 0:5] = np.log10(x[:, 0:5])
x[:, 6:8] = np.log10(x[:, 6:8])
x[x == -np.inf] = 0
x = np.abs(x)
x = normalize(x, norm='max', axis=0)

# exclude featurer
# x = np.delete(x, (5), axis=1)


title = ['delta ECB', 'theta ECB', 'alpha ECB', 'beta ECB', 'gamma ECB', 'Spindle AAE', 'EOG-SEM ECB', 'EOG-REM ECB',
         'EMG-Energy']

# run t-tests
k = 1
confidence = 0.99
for k in range(9):
    for i in range(6):
        for j in range(i + 1, 6):
            x1 = x[np.where(y == i)[0], k]
            x2 = x[np.where(y == j)[0], k]
            print("compare feature: ", title[k], "   stage", i, "and stage", j)
            ttest(x1, x2, confidence)

plt.figure(5)
plt.suptitle('EOG and EMG: Normalized Feature Distributions \n per Sleep Stages', size=12)
for j in range(9):
    i = j + 0
    p = 331
    plt.subplot(int(p + j))
    # plt.yscale('log')

    plt.title(title[i], size=10)
    plt.boxplot([x[np.where(y == 0)[0], i],
                 x[np.where(y == 1)[0], i],
                 x[np.where(y == 2)[0], i],
                 x[np.where(y == 3)[0], i],
                 x[np.where(y == 4)[0], i],
                 x[np.where(y == 5)[0], i]], 1, sym='')
    plt.xticks([1, 2, 3, 4, 5, 6], ['WAKE', 'S1', 'S2', 'S3', 'S4', 'REM'])
plt.show()

'''
label = [int(i) for i in y]

plt.figure(12)
plt.scatter(x[:, 8], x[:, 7], c=label)
plt.show()

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
