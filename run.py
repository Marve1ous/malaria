import numpy as np
from sklearn.metrics import roc_curve, auc, log_loss, average_precision_score, classification_report, confusion_matrix
from tensorflow.contrib.learn.python.learn.estimators._sklearn import accuracy_score

from evaluation import plot_confusion_matrix

from matplotlib import pyplot as plt
import load

param = {
    'image_size': 224,
    'classes': 2,
    'train': 'dataset/train',
    'validate': 'dataset/validate',
    'test': 'dataset/test',
    'batch_size': 32,
    'num_epoch': 10
}

names = ['custom', 'densenet121', 'mobilenetv2']


def choose_model(name):
    if name == names[0]:
        from custom_model import get_custom_model
        return get_custom_model(param['classes'])
    elif name == names[1]:
        from dense121 import get_densenet121_model
        param['batch_size'] = 4
        return get_densenet121_model(param['classes'])
    elif name == names[2]:
        from mobile import get_mobilev2_model
        return get_mobilev2_model(param['classes'])
    else:
        raise Exception("该模型不存在")


def train(name):
    model, checkpoint, tensorboard, pre_input, decode = choose_model(name)

    train_generator, validation_generator = load.data(train_path=param['train'],
                                                      vali_path=param['validate'],
                                                      size=param['image_size'],
                                                      batch_size=param['batch_size'],
                                                      preprocess_input=pre_input)

    hist = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=10,
        epochs=param['num_epoch'],
        validation_data=validation_generator,
        validation_steps=10,
        verbose=1,
        callbacks=[tensorboard, checkpoint]
    )

    return hist


def test(name, show_image=False):
    model, checkpoint, tensorboard, preprocess_input, decode = choose_model(name)

    x_image, x_label = load.load_data(path=param['test'],
                                      pre_input=preprocess_input,
                                      image_size=param['image_size'])

    pred = model.predict(
        x=x_image,
        batch_size=param['batch_size'],
        verbose=1
    )

    if show_image:
        for i in range(256):
            title = 'Predict class:' + str(np.argmax(pred[i]))
            plt.title(title)
            img = decode(x_image[i])
            plt.imshow(img)
            plt.show()

    return pred, x_label


def evaluate(hist, pred, truth):
    # compute the ROC-AUC values
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(param['classes']):
        fpr[i], tpr[i], _ = roc_curve(truth[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(truth.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(15, 10), dpi=300)
    lw = 1
    plt.plot(fpr[1], tpr[1], color='red',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics')
    plt.legend(loc="lower right")
    plt.show()

    # computhe the cross-entropy loss score
    score = log_loss(truth, pred)
    print(score)

    # compute the average precision score
    prec_score = average_precision_score(truth, pred)
    print(prec_score)

    # compute the accuracy on validation data
    test_accuracy = accuracy_score(truth.argmax(axis=-1), pred.argmax(axis=-1))
    print("Test_Accuracy = ", test_accuracy)

    # declare target names
    target_names = ['class 0(abnormal)', 'class 1(normal)']  # it should be normal and abnormal for linux machines

    # print classification report
    print(classification_report(truth.argmax(axis=-1), pred.argmax(axis=-1), target_names=target_names))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(truth.argmax(axis=-1), pred.argmax(axis=-1))
    np.set_printoptions(precision=4)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(15, 10), dpi=300)
    plot_confusion_matrix(cnf_matrix, classes=target_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    # transfer it back
    pred = np.argmax(pred, axis=1)
    truth = np.argmax(truth, axis=1)
    print(pred)
    print(truth)

    # visualizing losses and accuracy
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(param['num_epoch'])

    plt.figure(1, figsize=(15, 10), dpi=300)
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.style.use('classic')

    plt.figure(2, figsize=(15, 10), dpi=300)
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    plt.style.use('classic')
    plt.show()


if __name__ == '__main__':
    n = names[2]
    h = train(n)
    p, t = test(n)
    evaluate(h, p, t)
