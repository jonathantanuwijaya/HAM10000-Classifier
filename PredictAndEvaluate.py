import numpy as np
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, confusion_matrix


def evaluatemodel(model, test_batches, test_df, batch_size):
    predictions = model.predict(test_batches, steps=len(test_df) / batch_size, verbose=1)

    # geting predictions on test dataset
    y_pred = np.argmax(predictions, axis=1)
    targetnames = ['akiec', 'bcc', 'bkl', 'mel', 'nv']
    # getting the true labels per image
    y_true = test_batches.classes
    # getting the predicted labels per image
    y_prob = predictions
    from tensorflow.keras.utils import to_categorical
    y_test = to_categorical(y_true)

    # Creating classification report
    report = classification_report(y_true, y_pred, target_names=targetnames)

    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(y_test, predictions, labels=targetnames)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=targetnames)
    disp.plot()

    print("WEIGHTED")
    print("Precision: " + str(precision_score(y_true, y_pred, average='weighted')))
    print("Recall: " + str(recall_score(y_true, y_pred, average='weighted')))
    print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
    print("weighted Roc score: " + str(roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')) + "\n")

    print("MACRO")
    print("Precision: " + str(precision_score(y_true, y_pred, average='macro')))
    print("Recall: " + str(recall_score(y_true, y_pred, average='macro')))
    print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
    print("Macro Roc score: " + str(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')) + "\n")

    print('MICRO')
    print("Precision: " + str(precision_score(y_true, y_pred, average='micro')))
    print("Recall: " + str(recall_score(y_true, y_pred, average='micro')))
    print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
    tpr = {}
    fpr = {}
    roc_auc = {}
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print("Micro Roc score: " + str(roc_auc["micro"]))

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(7):
        r = roc_auc_score(y_test[:, i], y_prob[:, i])
        print("The ROC AUC score of " + targetnames[i] + " is: " + str(r))

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = dict()
    for i in range(7):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[0], tpr[0], 'v-', label='akiec: ROC curve of (area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], 'c', label='bcc: ROC curve of (area = %0.2f)' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], 'b', label='bkl: ROC curve of (area = %0.2f)' % roc_auc[2])
    plt.plot(fpr[3], tpr[3], 'g', label='df: ROC curve of (area = %0.2f)' % roc_auc[3])
    plt.plot(fpr[4], tpr[4], 'y', label='mel: ROC curve of (area = %0.2f)' % roc_auc[4])
    plt.plot(fpr[5], tpr[5], 'o-', label='nv: ROC curve of (area = %0.2f)' % roc_auc[5])
    plt.plot(fpr[6], tpr[6], 'r', label='vasc: ROC curve of (area = %0.2f)' % roc_auc[6])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of %s' % targetnames[i])
    plt.legend(loc="lower right")
    plt.show()
