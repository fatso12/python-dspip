import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, plot_confusion_matrix, roc_auc_score, roc_curve




iris = datasets.load_iris()
data_headers = iris.feature_names
print("the types of iris in this dataset are:")
print(list(iris.target_names))
print("the parameters we have are: ")
print(data_headers)
X = iris.data[:, :]
Y = iris.target
Y_vec=pd.DataFrame(Y)
X_vec=pd.DataFrame(X)
print(Y_vec[0].unique())
Y[Y <= 1] = 0
Y[Y == 2] = 1
train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.8)


neigh1 = KNeighborsClassifier(n_neighbors=3)
threshold = 0.5
neigh1.fit(train_x, train_y)
knn_predicted_proba_train = neigh1.predict_proba(train_x)
KNN_train_pred = (knn_predicted_proba_train [:,1] >= threshold).astype('int')
knn_predicted_proba_test = neigh1.predict_proba(test_x)
KNN_test_pred = (knn_predicted_proba_test [:,1] >= threshold).astype('int')
print(KNN_test_pred)
print("The Train Accuracy Of knn is: {0}%".format(metrics.accuracy_score(train_y, KNN_train_pred)*100))
print("The Test Accuracy Of knn is: {0}%".format(metrics.accuracy_score(test_y, KNN_test_pred)*100))
Train_confusion_matrix = metrics.confusion_matrix(train_y, KNN_train_pred)
Test_confusion_matrix = metrics.confusion_matrix(test_y, KNN_test_pred)
print("The Training Confusion matrix Of RandomForestClassifier is:\n{0}".format(Train_confusion_matrix))
print("The Testing Confusion matrix Of RandomForestClassifier is:\n{0}".format(Test_confusion_matrix))
plot_confusion_matrix(neigh1, test_x, test_y)
plt.show()


#Randrom forest
RandomForestClassifier = RandomForestClassifier(n_estimators=1000,max_features=3,min_samples_leaf=30,oob_score =True)
model=RandomForestClassifier.fit(train_x, train_y)
threshold = 0.5
predicted_proba_train = RandomForestClassifier.predict_proba(train_x)
RandomForestClassifier_train_predict = (predicted_proba_train [:,1] >= threshold).astype('int')
predicted_proba_test = RandomForestClassifier.predict_proba(test_x)
RandomForestClassifier_test_predict = (predicted_proba_test [:,1] >= threshold).astype('int')


print("The Train Accuracy Of RandomForestClassifier is: {0}%".format(metrics.accuracy_score(train_y, RandomForestClassifier_train_predict)*100))
print("The Test Accuracy Of RandomForestClassifier is: {0}%".format(metrics.accuracy_score(test_y, RandomForestClassifier_test_predict)*100))
Train_confusion_matrix = metrics.confusion_matrix(train_y, RandomForestClassifier_train_predict)
Test_confusion_matrix = metrics.confusion_matrix(test_y, RandomForestClassifier_test_predict)
print("The Training Confusion matrix Of RandomForestClassifier is:\n{0}".format(Train_confusion_matrix))
print("The Testing Confusion matrix Of RandomForestClassifier is:\n{0}".format(Test_confusion_matrix))
plot_confusion_matrix(RandomForestClassifier, test_x, test_y)
plt.show()


# roc curve and auc randomforest
ns_probs = [0 for _ in range(len(test_y))]
lr_probs = RandomForestClassifier.predict_proba(test_x)

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(test_y, ns_probs)
lr_auc = roc_auc_score(test_y, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Randomforest: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(test_y, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(test_y, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='randomforest')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()