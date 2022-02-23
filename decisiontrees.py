import numpy as np
import pandas as pd #for data franes 
import matplotlib.pyplot as plt # for data visualization 
import warnings
from sklearn.utils import shuffle
import statistics
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


class Node():
    def __init__(self, feature=None, ch0=None, ch1=None, ch2= None, target=None, isLeaf=False):
        self.feature = feature
        self.ch0 = ch0
        self.ch1 = ch1
        self.ch2 = ch2
        self.target = target
        self.isLeaf = isLeaf

def dataframe(data):
        return pd.read_csv(data)

def shuff(data):
        return shuffle(data)

def ttsplit(data):
        X, y = train_test_split(data, test_size=0.2, random_state=42) 
        return X,y

def entropy(target_col):
    val ,counts = np.unique(target_col,return_counts = True)
    entropy = 0
    for i in range(len(val)):
        entropy += (-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts))
    return entropy

def infogain(data, split_name, target_name = 'target'):
    total_entropy = entropy(data[target_name])
    vals,counts= np.unique(data[split_name],return_counts=True)
    average_entropy = 0
    for i in range(len(vals)):
        attribute =  data.where(data[split_name]==vals[i]).dropna()[target_name]
        average_entropy  += (counts[i]/ np.sum(counts))* entropy(attribute)
    return total_entropy - average_entropy

def unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

def decision_tree(data, attributes, label, parent=None):
    
    node = Node()
    if(unique(data['target'])==True):
        node.target = data['target'].value_counts().idxmax()
        node.isLeaf = True
        return node
    elif(len(attributes)==0):
        node.isLeaf = True
        node.target = data['target'].value_counts().idxmax()
        return node

    infogains = []
    for attribute in attributes:
        infogains.append(infogain(data, attribute, label))
    best = attributes[infogains.index(max(infogains))]
    new_attributes = []
    for attribute in attributes:
        if(attribute!=best):
            new_attributes.append(attribute)
    
    value, count = np.unique(data[best], return_counts= True)
    if len(value)!=3:
        node.isLeaf = True
        node.target = data['target'].value_counts().idxmax()
        return node
    else:
        node.feature = best
        node.ch0 = decision_tree(data[data[best] == 0], new_attributes, label)
        node.ch1 = decision_tree(data[data[best] == 1], new_attributes, label)
        node.ch2 = decision_tree(data[data[best] == 2], new_attributes, label)
        return node

def predict(test, node):
    if(node.isLeaf==True):
        return node.target
    else:
        branch = test[node.feature]
        if branch == 0:
            return predict(test, node.ch0)
        elif branch == 1:
            return predict(test, node.ch1)
        elif branch == 2:
            return predict(test, node.ch2)

def test(data,tree):
    #convert it to a dictionary
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    targets = data.iloc[:,-1:].to_dict(orient = "records")
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    count = 0
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree) 
        if(predicted.loc[i,"predicted"] == float(targets[i]['target'])):
            count+=1
    return count/len(data)

def training_plot(k):    
    accuracies = []
    label = 'target'
    attributes = k.columns[:-1]
    for i in range(100):
        X, y = ttsplit(shuff(k))
        tree = decision_tree(X, attributes, label)
        accuracies.append(test(X, tree))

    print("The average training accuracy is: ", sum(accuracies) / len(accuracies))
    print("The standard deviation of the training accuracy is: ", statistics.pstdev(accuracies))
    # Creating histogram
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(accuracies)

    # Show plot
    plt.show()

def testing_plot(k):    
    accuracies = []
    attributes = k.columns[:-1]
    label = 'target'
    for i in range(100):
        X, y = ttsplit(shuff(k))
        tree = decision_tree(X, attributes, label)
        accuracies.append(test(y, tree))

    print("The average of the testing accuracy is: ", sum(accuracies) / len(accuracies))
    print("The standard deviation of the testing accuracy is: ", statistics.pstdev(accuracies))
    # Creating histogram
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(accuracies)

    # Show plot
    plt.show()


data = 'votes.csv'
df = dataframe(data)
training_plot(df)
# testing_plot(df)



