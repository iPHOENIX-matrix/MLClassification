#Suvrajeet Jash
#20BEE0174

#Imbalanced Classification

from numpy import where
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot

#define dataset
X,y = make_classification(n_samples=1000,n_features=2,n_informative=2,n_redundant=0,n_classes=2,n_clusters_per_class=1,weights=[0.99,0.01],random_state=1)

#summarize dataset shape
print(X.shape,y.shape)

#summarize observations by class label
counter = Counter(y)
print(counter)

#summarize first few examples
for i in range(10):
    print(X[i],y[i])

#plot the dataset and color by class label
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix,0],X[row_ix,1],label=str(label))
pyplot.legend()
pyplot.show()


