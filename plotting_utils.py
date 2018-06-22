import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np


def save_figure_to_numpy(fig):
    """
    Saves a figure to a numpy array
    """
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram_to_numpy(spectrogram, rowsize=4, colsize=3):
    """
    Converts an array to a datatype to be plotted with tensorboard
    """
    fig, ax = plt.subplots(figsize=(rowsize, colsize))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_bincount(data, n_bins=100, title='Histogram', cmap=plt.cm.Blues,
                  filename='histogram.png'):
    """
    This function prints and plots the histogram.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(16, 16))
    hist = np.bincount(data, minlength=n_bins)
    plt.bar(range(n_bins), hist, align='center', width=0.7)
    plt.tight_layout()
    plt.ylabel('Count')
    plt.xlabel('Label')
    plt.xticks(range(n_bins), range(n_bins), rotation='vertical')
    plt.savefig(filename, bbox_inches='tight')


def plot_histogram(data, n_bins=100, title='Histogram', cmap=plt.cm.Blues,
                   force_xticks=False, filename='histogram.png'):
    """
    This function prints and plots the histogram.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(16, 16))
    hist, bins = np.histogram(data, bins=n_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.tight_layout()
    plt.ylabel('Count')
    plt.xlabel('Probability')
    plt.savefig(filename, bbox_inches='tight')


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues,
                          filename='conf_mat.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(32, 32))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename, bbox_inches='tight')


def reshape_to_matrix(samples, nrow, ncol):
    """
    This function reshapes a 3D batch of samples to a 2D matrix 
    of samples for plotting as a single image
    """
    dim = samples.shape[1]
    return (samples.reshape(nrow, ncol, dim, dim)
                   .transpose(0, 2, 1, 3)
                   .reshape(nrow*dim, ncol*dim))
