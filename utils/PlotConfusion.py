from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

labels = ['CV16', 'CVH550', 'CVH551', 'CVN68', 'CVN69',
          'CVN70', 'CVN71', 'CVN72', 'CVN73', 'CVN74',
          'CVN75', 'CVN76', 'CVN77', 'CVN78', 'L61',
          'R08', 'R22', 'R91', 'R911', 'R063']


def plot_figure(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def plot_confusion_matrix(truth_file, pred_file, image_name, net_name):
    y_true = np.loadtxt(truth_file)
    y_pred = np.loadtxt(pred_file)

    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    cls_num = len(labels)
    plt.figure(figsize=(cls_num, cls_num), dpi=300)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.00000001:
            plt.text(x_val, y_val, "%0.1f%%" % (c*100,), color='red', fontsize=10, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    prec = accuracy_score(y_true, y_pred)
    title_name = "%s (%.1f%%)"%(net_name, prec*100)
    plot_figure(cm_normalized, title=title_name)
    # show confusion matrix
    plt.savefig(image_name, format='png', transparent=False, dpi=300, bbox_inches="tight")
    #plt.show()
