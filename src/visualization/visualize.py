import matplotlib.pyplot as plt

def plot_history(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true_int, y_pred):
    labels = ['neutral', 'joy', 'surprise', 'anger', 'fear', 'disgust', 'sadness']
    cm_val = confusion_matrix(y_true_int, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt



def plot_precision_recall(y_true, y_pred):

    labels = ['neutral', 'joy', 'surprise', 'anger', 'fear', 'disgust', 'sadness']
    num_classes = len(labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for index, label in enumerate(labels)}

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        average_precision[i] = auc(recall[i], precision[i])

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], lw=2, label='Class {}: AP={:.2f}'.format(index_to_label[i], average_precision[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
