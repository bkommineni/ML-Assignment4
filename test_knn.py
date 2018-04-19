from knn import knn
from scipy.io import arff
import numpy as np
import pandas as pd

def loadDataSet(fileName):
    data, meta = arff.loadarff(fileName)
    df = pd.DataFrame(data=data)
    df = df.astype('int')
    return df.iloc[0:1082,0:9].values.tolist(), df.iloc[0:1082,9].values.tolist() , df.iloc[1082:1353,0:9].values.tolist(),  df.iloc[1082:1353,9].values.tolist()

def accuracy(labels, hypotheses):
    count = 0.0
    correct = 0.0

    for i in range(0,len(labels)):
        count += 1.0
        if labels[i] == hypotheses[i]:
            correct += 1.0
    return correct / count

X_train,Y_train,X_test,Y_test = loadDataSet("PhishingData.arff")



for i in range(2,33):
    clf = knn(k=i)
    clf.fit(X_train, Y_train)
    hypotheses = clf.predict(X_test)
    accuracy_score = accuracy(Y_test, hypotheses)
    print("k: "+str(i) + "  accuracy: " + str(accuracy_score))

