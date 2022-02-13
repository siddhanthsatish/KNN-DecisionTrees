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
                d.update({ dist : [ train_row['f1'], train_row['f2'], train_row['f3'], train_row['f4'], train_row['label'] ] })
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
    
    

            

            
    
obj = KNN(3)
data = 'iris.csv'
labels= ['f1', 'f2', 'f3', 'f4', 'label']
X, y = obj.train_test_split(obj.shuffle(obj.dataframe(data, labels)))
# print(X.head())
# print(y.head())
list = obj.knn(X, y)
acc = obj.accuracy(list)
print(acc)

time = []
acclist = []
for i in range(1, 51):
    obj = KNN(i)
    data = 'iris.csv'
    labels= ['f1', 'f2', 'f3', 'f4', 'label']
    X, y = obj.train_test_split(obj.shuffle(obj.dataframe(data, labels)))
    # print(X.head())
    # print(y.head())
    list = obj.knn(X, y)
    acc = obj.accuracy(list)
    time.append(i)
    acclist.append(acc)

plt.plot(time, acclist)
plt.show()









            




    
   