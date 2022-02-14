from json.tool import main
from mimetypes import init
import math
import numpy as np
import scipy.spatial
from collections import Counter
from collections import OrderedDict
import pandas as pd #for data franes 
import matplotlib.pyplot as plt # for data visualization 
import seaborn as sns # for data visualization
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class KNN:

    def __init__(self, k):
        self.k = k

    def dataframe(self, data, labels):
        return pd.read_csv(data, names = labels)
    
    def shuffle(self, data):
        return shuffle(data)

    def train_test_split(self, data):
        X, y = train_test_split(data, test_size=0.2)
        return X,y

    def knn(self, X, y):      
        d = {}
        knn = []
        for test_index, test_row in y.iterrows():
            d = {}
            point1 = np.array([test_row['f1'], test_row['f2'], test_row['f3'], test_row['f4']])
            for train_index, train_row in X.iterrows():
                dist = 0
                point2 = np.array([train_row['f1'], train_row['f2'], train_row['f3'], train_row['f4']])
                dist = scipy.spatial.distance.euclidean(point1, point2)
                d.update({ dist : [ train_row['f1'], train_row['f2'], train_row['f3'], train_row['f4'], train_row['label'] ] }) #make tuple
            count=0
            knearest = []
            for i in sorted(d.keys()):
                if(count<self.k):
                    knearest.append(d[i][4])
                    count+=1
                else:
                    break
            # print((knearest, test_row['label']))
            knn.append((knearest, test_row['label']))
        # print(knn)
        return knn
    

    def accuracy(self, knn):
        count = 0 
        for tup in knn:
            l = tup[0]
            occurence_count = Counter(l)
            s = occurence_count.most_common(1)[0][0]
            if s == tup[1]:
                count+=1
        return count/len(knn)

            
    
obj = KNN(1)
data = 'iris.csv'
labels= ['f1', 'f2', 'f3', 'f4', 'label']
X, y = obj.train_test_split(obj.shuffle(obj.dataframe(data, labels)))
list = obj.knn(X, X)
acc = obj.accuracy(list)
print(acc)
count = 1
accd = {}
for i in range(1, 51, 2):
    accd.update( {i: []} )
print(accd)

#change object oriented design and value for j
#shuffle before
#normalise values
#graph for training and testing

j=1
#training accuracy for 1 to 51 values
for i in range(1, 2):
    obj = KNN(j)
    data = 'iris.csv'
    labels= ['f1', 'f2', 'f3', 'f4', 'label']
    X, y = obj.train_test_split(obj.shuffle(obj.dataframe(data, labels)))
    for j in range(1, 51, 2):
        list = obj.knn(X, X)
        acc = obj.accuracy(list)
        accd[j].append(acc)
        count+=1
print(accd)

#finiding mean accuracies
time = []
mean_acc = []
c = 1
for i in accd:
    mean_acc.append(sum(accd[i])/len(accd[i]))
    time.append(c)
    c+=2
plt.plot(time, mean_acc)
plt.show()

j=1
#testing accuracy for 1 to 51 values
for i in range(1, 2):
    obj = KNN(j)
    data = 'iris.csv'
    labels= ['f1', 'f2', 'f3', 'f4', 'label']
    X, y = obj.train_test_split(obj.shuffle(obj.dataframe(data, labels)))
    for j in range(1, 51, 2):
        list = obj.knn(X, y)
        acc = obj.accuracy(list)
        accd[j].append(acc)
        count+=1
print(accd)

#finiding mean accuracies
time = []
mean_acc = []
c = 1
for i in accd:
    mean_acc.append(sum(accd[i])/len(accd[i]))
    time.append(c)
    c+=2
plt.plot(time, mean_acc)
plt.show()












            




    
   