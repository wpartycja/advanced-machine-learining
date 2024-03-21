import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,  plot_tree
from sklearn.model_selection import train_test_split
import os

os.chdir("/home/wpartycja/mgr/advanced_machine_learning/04-trees/data")

# TASK 1:
    
# Read data set:    
data_1 = pd.read_csv("SAheart.data", index_col=0)
data_1.head()
X = data_1.iloc[:, :-1]
y = data_1.iloc[:, -1]
X = X.replace("Present", 1)
X = X.replace("Absent", 0)
X.head()

# Train-test split:    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

#Fit the model:
#clf = DecisionTreeClassifier(criterion='entropy',max_depth=3)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

#Drwa a structure:
plt.figure(1,figsize=(20,10))
plot_tree(clf)
plt.show()

#Cost-complexity prunning:
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)


train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()

