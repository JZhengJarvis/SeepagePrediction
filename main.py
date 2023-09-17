# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import GRA
from sklearn.svm import SVR

# create sample data
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel()

# add noise to the target
y[::5] += 3 * (0.5 - np.random.rand(20))

#print(GRA.grey_correlation_analysis(X,y))

# create and train the SVR model
model = SVR(kernel='rbf', C=1e3, gamma=0.1)
model.fit(X, y)

# predict on new data
X_test = np.linspace(0, 5, 100)[:, np.newaxis]
y_pred = model.predict(X_test)

# plot the results
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X_test, y_pred, color='navy', label='SVR')
plt.legend()
plt.show()
