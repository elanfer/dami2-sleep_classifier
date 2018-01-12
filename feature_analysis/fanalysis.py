# read data
data_array = genfromtxt('../data/Features_SC.txt', delimiter=',')
end = np.asarray(data_array).shape[1]
y = np.asarray(data_array[:, 3])
x = np.asarray(data_array[:, 4:end])

# plot:
