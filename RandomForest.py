from statistics import mode
import vocabulary 
import numpy as np 
import math
from sklearn.model_selection import learning_curve
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt



def custom_learning_curve(x_train, y_train, x_test, y_test, n_splits):
  
    split_size = int(len(x_train) / n_splits)
    x_splits = np.split(x_train, n_splits) 
    y_splits = np.split(y_train, n_splits)

    train_accuracies = list()
    test_accuracies = list()
    curr_x = x_splits[0]
    print(curr_x.shape)
    curr_y = y_splits[0]
    print(curr_y.shape)

    train_precision = list()
    test_precision = list()
    train_recall = list()
    test_recall = list()
    train_f1 = list()
    test_f1 = list()

    clf = RandomForest(n_trees=5, max_depth=5)
    clf.fit(X_train, y_train)

    train_accuracies.append(accuracy_score(curr_y, clf.predict(curr_x)))
    test_accuracies.append(accuracy_score(y_test, clf.predict(x_test)))
    train_precision.append(precision_score(curr_y, clf.predict(curr_x)))
    test_precision.append(precision_score(y_test, clf.predict(x_test)))
    train_recall.append(recall_score(curr_y, clf.predict(curr_x)))
    test_recall.append(recall_score(y_test, clf.predict(x_test)))
    train_f1.append(f1_score(curr_y, clf.predict(curr_x)))
    test_f1.append(f1_score(y_test, clf.predict(x_test)))

    

    for i in range(1, len(x_splits)):
        
        curr_x = np.concatenate((curr_x, x_splits[i]), axis=0)
        print(curr_x.shape)
        curr_y = np.concatenate((curr_y, y_splits[i]), axis=0)
        print(curr_y.shape)
        clf.fit(curr_x, curr_y)

        

        train_accuracies.append(accuracy_score(curr_y, clf.predict(curr_x)))
        test_accuracies.append(accuracy_score(y_test, clf.predict(x_test)))
        train_precision.append(precision_score(curr_y, clf.predict(curr_x)))
        test_precision.append(precision_score(y_test, clf.predict(x_test)))
        train_recall.append(recall_score(curr_y, clf.predict(curr_x)))
        test_recall.append(recall_score(y_test, clf.predict(x_test)))
        train_f1.append(f1_score(curr_y, clf.predict(curr_x)))
        test_f1.append(f1_score(y_test, clf.predict(x_test)))
        

    plt.plot(list(range(split_size, len(x_train) + split_size, split_size)), train_accuracies, 'o-', color="b", label="Training accuracy")
    plt.plot(list(range(split_size, len(x_train) + split_size, split_size)), test_accuracies, 'o-', color="r",label="Testing accuracy")
    plt.legend(loc="lower right")
    plt.xlabel('Data size')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(list(range(split_size, len(x_train) + split_size, split_size)), train_precision, 'o-', color="b", label="Training precision")
    plt.plot(list(range(split_size, len(x_train) + split_size, split_size)), test_precision, 'o-', color="r",label="Testing precision")
    plt.legend(loc="lower right")
    plt.xlabel('Data size')
    plt.ylabel('Precision')
    plt.show()

    plt.plot(list(range(split_size, len(x_train) + split_size, split_size)), train_recall, 'o-', color="b", label="Training recall")
    plt.plot(list(range(split_size, len(x_train) + split_size, split_size)), test_recall, 'o-', color="r",label="Testing recall")
    plt.legend(loc="lower right")
    plt.xlabel('Data size')
    plt.ylabel('Recall')
   
    plt.show()

    plt.plot(list(range(split_size, len(x_train) + split_size, split_size)), train_f1, 'o-', color="b", label="Training f1")
    plt.plot(list(range(split_size, len(x_train) + split_size, split_size)), test_f1, 'o-', color="r",label="Testing f1")
    plt.legend(loc="lower right")
    plt.xlabel('Data size')
    plt.ylabel('f1')
    plt.show()
    

    print(classification_report(y_train, clf.predict(X_train),
                             zero_division=1))
    print(classification_report(y_test, clf.predict(X_test),
                             zero_division=1))
class Node: 
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category
        

class ID3:
    def __init__(self, features, max_depth=None):
        self.tree = None
        self.features = features
        self.max_depth = max_depth

    def fit(self, x, y):
        most_common = mode(y.flatten())
        self.tree = self.create_tree(x, y, features=np.arange(len(self.features)), category=most_common, depth=0)
        return self.tree

    def create_tree(self, x_train, y_train, features, category, depth):
        
        
        if len(x_train) == 0:
            return Node(checking_feature=None, is_leaf=True, category=category)
        
        if np.all(y_train.flatten() == 0):
            return Node(checking_feature=None, is_leaf=True, category=0)
        elif np.all(y_train.flatten() == 1):
            return Node(checking_feature=None, is_leaf=True, category=1)
        
        if len(features) == 0 or (self.max_depth is not None and depth == self.max_depth):
            return Node(checking_feature=None, is_leaf=True, category=mode(y_train.flatten()))

        igs = list()
        for feat_index in features.flatten():
            igs.append(self.calculate_ig(y_train.flatten(), [example[feat_index] for example in x_train]))

        max_ig_idx = np.argmax(np.array(igs).flatten())
        m = mode(y_train.flatten())

        root = Node(checking_feature=max_ig_idx)

        x_train_0 = x_train[x_train[:, max_ig_idx] == 0, :]
        y_train_0 = y_train[x_train[:, max_ig_idx] == 0].flatten()

        x_train_1 = x_train[x_train[:, max_ig_idx] == 1, :]
        y_train_1 = y_train[x_train[:, max_ig_idx] == 1].flatten()

        new_features_indices = np.delete(features.flatten(), max_ig_idx)

        root.left_child = self.create_tree(x_train=x_train_1, y_train=y_train_1, features=new_features_indices, 
                                           category=m, depth=depth + 1)
        
        root.right_child = self.create_tree(x_train=x_train_0, y_train=y_train_0, features=new_features_indices,
                                            category=m, depth=depth + 1)
        
        return root

    '''calculate_ig method: This method calculates the information gain 
for a given feature by computing the entropy of the classes 
before and after the split.'''

    @staticmethod
    def calculate_ig(classes_vector, feature):
        classes = set(classes_vector)

        HC = 0
        for c in classes:
            PC = list(classes_vector).count(c) / len(classes_vector)  # P(C=c)
            HC += - PC * math.log(PC, 2)  # H(C)
            # print('Overall Entropy:', HC)  # entropy for C variable
            
        feature_values = set(feature)  # 0 or 1 in this example
        HC_feature = 0
        for value in feature_values:
            # pf --> P(X=x)
            pf = list(feature).count(value) / len(feature)  # count occurences of value 
            indices = [i for i in range(len(feature)) if feature[i] == value]  # rows (examples) that have X=x

            classes_of_feat = [classes_vector[i] for i in indices]  # category of examples listed in indices above
            for c in classes:
                # pcf --> P(C=c|X=x)
                pcf = classes_of_feat.count(c) / len(classes_of_feat)  # given X=x, count C
                if pcf != 0: 
                    # - P(X=x) * P(C=c|X=x) * log2(P(C=c|X=x))
                    temp_H = - pf * pcf * math.log(pcf, 2)
                    # sum for all values of C (class) and X (values of specific feature)
                    HC_feature += temp_H
        
        ig = HC - HC_feature
        return ig    

        
    '''predict method: This method uses the trained decision tree to predict the category for a given set of examples.'''
    def predict(self, x):
        predicted_classes = list()

        for unlabeled in x:  # for every example 
            tmp = self.tree  # begin at root
            while not tmp.is_leaf:
                if unlabeled.flatten()[tmp.checking_feature] == 1:
                    tmp = tmp.left_child
                else:
                    tmp = tmp.right_child
            
            predicted_classes.append(tmp.category)
        
        return np.array(predicted_classes)


class RandomForest:
    def __init__(self, n_trees, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, x, y):
        for _ in range(self.n_trees):
            # Create a bootstrap sample of the training data
            x_sample, y_sample = resample(x, y)

            # Create an ID3 tree and fit it on the bootstrap sample
            id3_tree = ID3(features=np.arange(x.shape[1]), max_depth=self.max_depth)
            id3_tree.fit(x_sample, y_sample)

            # Add the trained tree to the list of trees in the forest
            self.trees.append(id3_tree)

    def predict(self, x):
        # Make predictions using each tree in the forest
        predictions = [tree.predict(x) for tree in self.trees]

        # Combine predictions using majority voting
        ensemble_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return ensemble_predictions


print("voc")    
m=1000
n=50
k=50
X_train, y_train, X_test, y_test = vocabulary.voc(m, n, k)

custom_learning_curve(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test, n_splits=2)
# def fn():
#     numbers= [1,5,10,30,50,100]
#     for i in numbers:
#         clf = RandomForest(n_trees=i, max_depth=5)
#         clf.fit(X_train, y_train)
        
#         print("Tree "+ str(i))
#         print(accuracy_score(y_train, clf.predict(X_train)))
#         print(accuracy_score(y_test, clf.predict(X_test)))
        
# fn()