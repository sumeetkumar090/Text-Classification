#Task 1 - Sumeet Kumar -  5873137 - Sk521

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import itertools
import scipy.sparse as sps

#Confusion Matrix function
def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure(figsize = (20,10)) 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid('false')
   
os.chdir("D:/UOW/Sem3/Big Data Analytics/Ass2/20news-bydate-matlab/20news-bydate/matlab")

#Read the data
trainingData = pd.read_table('train.data',sep=" ",names=['docId', 'wordId', 'count'])
testingData = pd.read_table('test.data',sep=" ",names=['docId', 'wordId', 'count'])
traininglabel = pd.read_table('train.label',sep=" ",names=['labelId'])
testinglabel = pd.read_table('test.label',sep=" ",names=['labelId'])
trainingmap = pd.read_table('train.map',sep=" ",names=['labelName','labelId'])
testingmap = pd.read_table('test.map',sep=" ",names=['labelName','labelId'])
trainingData

#Document matrix for testing and training
mat = sps.coo_matrix((trainingData["count"].values, (trainingData["docId"].values-1, trainingData["wordId"].values-1)))
training_data_matrix = mat.tocsc()
training_data_matrix.shape 

test_mat  = sps.coo_matrix((testingData["count"].values, (testingData["docId"].values-1, testingData["wordId"].values-1)))
testing_data_matrix = test_mat.tocsc()
testing_data_matrix = testing_data_matrix[:,:training_data_matrix.shape[1]] 

#Naive Bayes classifier
naiveBayesclassifier = MultinomialNB(alpha=.01, class_prior=None, fit_prior=True)
naiveBayesclassifier.fit(training_data_matrix, traininglabel["labelId"])
#print(naiveBayesclassifier)

# Predict the test results
prediction = naiveBayesclassifier.predict(testing_data_matrix)

#Printing accuracy
print("Accuracy = {0}".format(metrics.f1_score(testinglabel["labelId"], prediction, average='macro')* 100))

#Confusion Matrix
cm = confusion_matrix(testinglabel["labelId"], prediction)
plot_confusion_matrix(cm, classes=trainingmap["labelName"],title='Confusion matrix')

n_classes = 20
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(20):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(testinglabel["labelId"]))[:, i], np.array(pd.get_dummies(prediction))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# Plot all ROC curves
plt.figure(figsize=(20,10))

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class - {0} (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for 20 NewsGroups')
plt.legend(loc="lower right")
plt.show()