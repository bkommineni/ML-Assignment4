from classifier import classifier
from numpy import *

class knn(classifier):
    def __init__(self, k=3):
        self.k = k

    def fit(self, Xin, Yin):
        self.data = Xin
        self.label = Yin
        pass

    def predict(self, X):
        hypotheses = []
        for i in range(0,len(X)):
            test_data = X[i]
            predict_class = self.predict_helper(test_data)
            hypotheses.append(predict_class)
        return hypotheses

    def distance(self, training_data, test_data):
        multiples = [math.pow(b-a,2) for a, b in zip(training_data, test_data)]
        return math.sqrt(sum(multiples))

    def predict_helper(self, test_data):
        dist = []
        for i in range(0,len(self.data)):
            dist.append((self.distance(self.data[i], test_data),self.label[i]))

        sorted_dist = sorted(dist,key=lambda x:x[0])
        neighbours = sorted_dist[:self.k]
        neighbours_classes = [x[1] for x in neighbours]
        classes = []
        for i in range(0,len(neighbours_classes)):
            if neighbours_classes[i] not in classes:
                classes.append(neighbours_classes[i])
        classes_vals = dict()
        for class_i in neighbours_classes:
            if class_i in classes_vals:
                classes_vals[class_i] = classes_vals.get(class_i) + 1
            else:
                classes_vals[class_i] = 1

        maximum_class = max(classes_vals, key=classes_vals.get)
        return maximum_class