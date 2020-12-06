import matplotlib.pyplot as plt
import numpy as np

def generate_loss_plot(data_dir=''):
    loss = []
    f = open(data_dir + 'log.txt', 'r', encoding='utf8')
    for line in f.readlines():
        line = line.split()[1]
        loss.append(float(line))
    plt.figure(figsize=(13, 13))
    plt.title('Loss plot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot([(i + 1) for i in range(len(loss))], loss)
    plt.savefig(data_dir + 'loss.png')

def generate_accuracy_plot(data_dir=''):
    accuracy = []
    f = open(data_dir + 'log.txt', 'r', encoding='utf8')
    for line in f.readlines():
        line = line.split()[0]
        accuracy.append(float(line) * 100)
    plt.figure(figsize=(13, 13))
    plt.title('Accuracy plot')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot([(i + 1) for i in range(len(accuracy))], accuracy)
    plt.savefig(data_dir + 'accuracy.png')

def generate_confusion_matrix_plot(conf_mat, classes, normalize=False, title='', data_dir=''):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(13, 13))
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(conf_mat.shape[1]),
           yticks=np.arange(conf_mat.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True labels',
           xlabel='Predicted labels')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, format(conf_mat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_mat[i, j] > thresh else "black")
    fig.tight_layout()
    if normalize:
        plt.savefig(data_dir + 'confusion matrix normalized.png')
    else:
        plt.savefig(data_dir + 'confusion matrix.png')


