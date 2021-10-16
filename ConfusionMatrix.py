from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
import numpy as np

label = ['mel', 'akiec', 'bcc', 'nv', 'bkl']


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm2 = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    cm2 = cm2 * 100
    print(cm)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm2, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm2.shape[1]),
           yticks=np.arange(cm2.shape[0]),
           xticklabels=label, yticklabels=label,
           title=title,
           ylabel='Predicted label',
           xlabel='True label')

    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            z = cm[i, j]
            z = np.around(z)

            ax.text(j, i, format(z),
                    ha="center", va="center",
                    color="white" if i == j else "black")
    fig.tight_layout()
    return ax
