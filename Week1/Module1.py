import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from adspy_shared_utilities import plot_fruit_knn

fruits = pd.read_table("fruit_data_with_colors.txt")

lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print (lookup_fruit_name)

X = fruits[['mass','width','height']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
fig = plt.figure(1)
cmap = cm.get_cmap('gnuplot')
#scatter = pd.scatter_matrix(X_train, c = y_train, marker = 'o', S=40, hist_kwds = {'bins':15}, figsize = (12,12), cmap = cmap)

fig = plt.figure(2)
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['mass'], X_train['width'], X_train['height'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('mass')
ax.set_ylabel('width')
ax.set_zlabel('height')

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
print (knn.score(X_test, y_test))

fruit_prediction = knn.predict([[20, 4.3, 5.5]])
print (lookup_fruit_name[fruit_prediction[0]])

fruit_prediction = knn.predict([[100, 6.3, 8.5]])
print (lookup_fruit_name[fruit_prediction[0]])

plot_fruit_knn(X_train, y_train, 1, 'uniform')
plot_fruit_knn(X_train, y_train, 5, 'uniform')
plot_fruit_knn(X_train, y_train, 10, 'uniform')

k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    scores.append(knn.score(X_test, y_test))
    
plt.figure(3)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])
plt.show()
    
