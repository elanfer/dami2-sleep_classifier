import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import stats
from sklearn.preprocessing import normalize


def ttest(x1, x2, confidence=0.95):
    # return: statistic, p-value
    ind_t_test = stats.ttest_ind(x1, x2)

    return ind_t_test[0], ind_t_test[1]





# read data
data_array = np.genfromtxt('../data/Features_SC.txt', delimiter=',')
end = np.asarray(data_array).shape[1]
y = np.asarray(data_array[:, 3])
x = np.asarray(data_array[:, 4:end])

# normalization:
x = np.abs(x)
ten = np.zeros(x.shape[0]) + 10
# x[:, 0:5] = np.log10(x[:, 0:5])
# x[:, 6:8] = np.log10(x[:, 6:8])
x[x == -np.inf] = 0
x = np.abs(x)
x = normalize(x, norm='max', axis=0)

x[:, 0] = 5.0 * x[:, 0]
x[:, 1] = 9.0 * x[:, 1]
x[:, 2] = 26.5 * x[:, 2]
x[:, 3] = 45.5 * x[:, 3]
x[:, 4] = 50. * x[:, 4]
x[:, 5] = 4.8 * x[:, 5]
x[:, 6] = 22. * x[:, 6]
x[:, 7] = 9. * x[:, 7]
x[:, 8] = 115. * x[:, 8]
# exclude featurer
# x = np.delete(x, (5), axis=1)


title = ['delta ECB', 'theta ECB', 'alpha ECB', 'beta ECB', 'gamma ECB', 'Spindle AAE', 'EOG-SEM ECB', 'EOG-REM ECB',
         'EMG-Energy']

# One-way ANOVA

print(scipy.stats.f_oneway(y, x[:, 1]))



# run t-tests
k = 1
confidence = 0.995
for k in range(9):
    t = np.zeros([6, 6])
    p = np.zeros([6, 6], dtype=bool)
    print
    for i in range(6):
        for j in range(6):
            x1 = x[np.where(y == i)[0], k]
            x2 = x[np.where(y == j)[0], k]
            t[i, j], px = ttest(x1, x2, confidence)
            p[i, j] = 1 - px > confidence
    # print('t-Matrix for feature:', title[k] )
    # print(t)
    print('significant dif. Matrix for feature:', title[k])
    print(p)

plt.figure(5)
plt.suptitle('Normalized Feature Distributions per Sleep Stages', size=12)
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
