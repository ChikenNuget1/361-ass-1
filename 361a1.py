
# Import libraries
# Pandas to read the csv
# Numpy for math-related functinos
import pandas as pd
import numpy as np


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# Set a random seed to make the experiment reproducible
np.random.seed(44)

# Read the csv data and split by line and then split each line by ;
data = pd.read_csv('winequality-red.csv', sep=';')

"""
Change the quality column to binary instead of numeric.
1 == Good
0 == Bad

I chose to split at 6 as it gives an intuitive split on bad vs good for a rating system out of 10.
With quality < 6 being bad while quality >= 6 being good
"""
data['quality'] = (data['quality'] >= 6).astype(int)

# I chose an 80/20 split with 80% of the data being used for training and 20% of the data being used for testing/validation
# EXPLAIN WHY YOU CHOSE 80/20 SPLIT
train = data.sample(frac=0.8, random_state=44)
test = data.drop(train.index)

# We seperate the features and the labels
x_train = train.drop("quality", axis=1)
y_train = train["quality"]

x_test = test.drop("quality", axis=1)
y_test = test["quality"]

# I can then move on to implementing the entropy function and then the information gain function.
# EXPLAIN MORE AND GO INTO DETAIL OF WHAT EACH FUNCTION DOES
def entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    prob = counts/counts.sum()

    return -np.sum(prob * np.log2(prob))


# Information gain measures how good a split is
def information_gain(parent, left, right):
    parent_entropy = entropy(parent)
    
    n = len(parent)
    n_left = len(left)
    n_right = len(right)

    # Calculate weighted entropy of both left and right child
    weighted_entropy = (
        (n_left / n) * entropy(left) +
        (n_right / n) * entropy(right)
    )

    return parent_entropy - weighted_entropy


# Search for best feature and threshold to split on
def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature in X.columns:
        thresholds = X[feature].unique()

        for threshold in thresholds:

            left_mask = X[feature] <= threshold
            right_mask = X[feature] > threshold

            if sum(left_mask) == 0 or sum(right_mask) == 0:
                continue

            gain = information_gain(
                y, y[left_mask], y[right_mask]
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


def train_tree(X, y, depth=0, stopping_depth=None):

    # Stopping condition
    if len(np.unique(y)) == 1:
        return Node(value=y.iloc[0])
    
    if stopping_depth is not None and depth >= stopping_depth:
        return Node(value=y.mode()[0])
    
    feature, threshold = best_split(X, y)

    if feature is None:
        return Node(value=y.mode()[0])

    left_mask = X[feature] <= threshold
    right_mask = X[feature] > threshold

    # A check to prevent infinite recursion
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return Node(value=y.mode()[0])

    left_child = train_tree(
        X[left_mask], y[left_mask], depth+1, stopping_depth
    )

    right_child = train_tree(
        X[right_mask], y[right_mask], depth+1, stopping_depth
    )

    return Node(feature, threshold, left_child, right_child)


tree = train_tree(x_train, y_train)

# Depth control 
# Discuss these trees in the report
tree2 = train_tree(x_train, y_train, stopping_depth=2)
tree3 = train_tree(x_train, y_train, stopping_depth=3)
tree4 = train_tree(x_train, y_train, stopping_depth=4)


def predict_sample(node, sample):
    if node.value is not None:
        return node.value
    
    if sample[node.feature] <= node.threshold:
        return predict_sample(node.left, sample)
    else:
        return predict_sample(node.right, sample)
    

def predict(tree, X):
    predictions = []

    for _, row in X.iterrows():
        predictions.append(predict_sample(tree, row))

    return np.array(predictions)


predictions=predict(tree, x_test)

accuracy = np.mean(predictions == y_test)

print("accuracy: ", accuracy)
