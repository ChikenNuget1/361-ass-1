
# Import libraries
# Pandas to read the csv
# Numpy for math-related functinos
import pandas as pd
import numpy as np

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