from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold
from ConfusionMatrix import plot_confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


def KFoldtrain(num_folds, inputs, targets, model, categories):
    kfold = KFold(n_splits=num_folds, random_state=3, shuffle=True)
    fold_no = 1
    lr = 0.0001
    i = 1
    j = 0

    akurasi_per_fold = []
    prec_per_fold = []
    rec_per_fold = []
    f1_per_fold = []
    learningrate = []
    acc_per_fold = []
    loss_per_fold = []
    checkpoint = ModelCheckpoint(filepath='model/HAMResNet152.hdf5', monitor='val_accuracy', save_best_only=True,
                                 save_weights_only=True)
    Earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=40, min_delta=0.001)

    class_weights = {
        0: 1.0,  # bkl
        1: 1.0,  # nv
        2: 1.0,  # mel
        3: 1.0,  # bcc
        4: 5.0,  # akiec
    }

    for train, test in kfold.split(inputs, targets):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        r = model.fit(inputs[train], targets[train], validation_data=(inputs[test], targets[test]),
                      verbose=1, class_weight=class_weights, callbacks=[checkpoint, Earlystop], steps_per_epoch=10,
                      epochs=100)

        scores = model.evaluate(inputs[test], targets[test], verbose=0)

        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        Y_pred = model.predict(inputs[test], 30)
        y_pred = np.argmax(Y_pred, axis=1)
        plot_confusion_matrix(y_pred, targets[test], classes=categories, normalize=True, title='Confusion Matrix')
        # cm = confusion_matrix(y_pred,targets[test])
        accscore = accuracy_score(y_pred, targets[test])
        presscore = precision_score(y_pred, targets[test], average=None)
        recscore = recall_score(y_pred, targets[test], average=None)
        f1score = f1_score(y_pred, targets[test], average=None)

        akurasi_per_fold.append(np.mean(accscore))
        prec_per_fold.append(np.mean(presscore))
        rec_per_fold.append(np.mean(recscore))
        f1_per_fold.append(np.mean(f1score))

        plt.savefig('Kfold-' + str(i))
        plt.show()
        print(classification_report(targets[test], y_pred))

        fold_no += 1

    print('rata-rata acc pada learning rate ', lr, ' adalah ', np.mean(akurasi_per_fold))
    # print('rata-rata loss pada learning rate ', lr, 'adalah', losslr)
    print('rata-rata presicion pada learning rate ', lr, 'adalah', np.mean(prec_per_fold))
    print('rata-rata recall pada learning rate ', lr, 'adalah', np.mean(rec_per_fold))
    print('rata-rata F1 Score pada learning rate ', lr, 'adalah', np.mean(f1_per_fold))

    learningrate.append(lr)
    lr += 0.0111
    i += 1
    j += 1
    fold_no = 1
    model.save('my_model.h5')
    print('=========================================================================')
