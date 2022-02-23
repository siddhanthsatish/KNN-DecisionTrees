from json.tool import main
from mimetypes import init
import math
import numpy as np
import scipy.spatial
from collections import Counter
import pandas as pd #for data franes 
import matplotlib.pyplot as plt # for data visualization 
import warnings
from sklearn.utils import shuffle
import statistics
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

#converting into dataframe
def dataframe(data, labels):
        return pd.read_csv(data, names = labels)

#shuffling the datframe
def shuff(data):
        return shuffle(data)

#80-20 split on the data frame
def ttsplit(data):
        X, y = train_test_split(data, test_size=0.2)
        return X,y

#normalising
def minmaxscaling(column) :
    return ((column - column.min()) / (column.max() - column.min()))

#normalising all columns
def normalize(df):
    for col in df.columns[0:4]:
        df[col] = minmaxscaling(df[col])
    return df

#sortinng the tupple
def sort_tuple(tup): 
    tup.sort(key = lambda x: x[0]) 
    return tup 

#knn algorithm
def knn(X, y, k):      
        d = []
        knn = []
        #iterating through test set points
        for test_index, test_row in y.iterrows():
            d = []
            point1 = np.array([test_row['f1'], test_row['f2'], test_row['f3'], test_row['f4']])
            #iterating through train set points
            for train_index, train_row in X.iterrows():
                dist = 0
                point2 = np.array([train_row['f1'], train_row['f2'], train_row['f3'], train_row['f4']])
                dist = scipy.spatial.distance.euclidean(point1, point2)
                d.append( (dist, [ train_row['f1'], train_row['f2'], train_row['f3'], train_row['f4'], train_row['label'] ]) ) 
            count=0
            knearest = []
            new_d = sort_tuple(d)
            #selecting k-nearest points
            for i in new_d:
                if(count<k):
                    knearest.append(i[1][4])
                    count+=1
                else:
                    break
            #storing a record of predicted points and actual points
            knn.append((knearest, test_row['label']))
        return knn

#overall accuracy
def accuracy(knn):
        count = 0 
        for tup in knn:
            l = tup[0] #k nearest points for a particular point
            occurence_count = Counter(l)
            s = occurence_count.most_common(1)[0][0] #getting the most common label among them
            if s == tup[1]: #check if prediction and actual are equal
                count+=1
        return count/len(knn) #total accuracy

def plot_training_accuracy():
    
    accd = {}
    for i in range(1, 51, 2):
        accd.update( {i: []} )
    

    #training accuracy for 1 to 51 values
    for i in range(1, 21):
            data = 'iris.csv'
            labels= ['f1', 'f2', 'f3', 'f4', 'label']
            X, y = ttsplit(shuff(dataframe(data, labels)))
            for k in range(1, 51, 2):
                list = knn(X, X, k)
                acc = accuracy(list)
                accd[k].append(acc)
    

    #finiding mean and std of training accuracies
    time = []
    mean_acc = []
    c = 1
    std_acc = []
    for i in accd:
        mean_acc.append(sum(accd[i])/len(accd[i]))
        std_acc.append(statistics.pstdev(accd[i]))
        time.append(c)
        c+=2

    print("The mean value of k for values 1 to 51 for the testing set are: ", mean_acc)
    print("The standard deviation value of k for values 1 to 51 for the testing set are: ", std_acc)
    
    plt.errorbar(time, mean_acc, yerr = std_acc, label ='Mean Training Accuracies')
    plt.show()

def plot_testing_accuracy():

    accd = {}
    for i in range(1, 51, 2):
        accd.update( {i: []} )

    #testing accuracy for 1 to 51 values
    for i in range(1, 21):
            data = 'iris.csv'
            labels= ['f1', 'f2', 'f3', 'f4', 'label']
            X, y = ttsplit(shuff(dataframe(data, labels)))
            for k in range(1, 51, 2):
                list = knn(X, y, k)
                acc = accuracy(list)
                accd[k].append(acc)

    #finiding mean and std of testing accuracies
    time = []
    mean_acc = []
    c = 1
    std_acc = []
    for i in accd:
        mean_acc.append(sum(accd[i])/len(accd[i]))
        std_acc.append(statistics.pstdev(accd[i]))
        time.append(c)
        c+=2

    print("The mean value of k for values 1 to 51 for the training set are: ", mean_acc)
    print("The standard deviation value of k for values 1 to 51 for the training set are: ", std_acc)

    plt.errorbar(time, mean_acc, yerr = std_acc, label ='Mean Testing Accuracies')
    plt.show()


plot_training_accuracy()
plot_testing_accuracy()












