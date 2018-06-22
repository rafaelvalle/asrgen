import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np

speaker_id = 10
n_classes = 103
name = 'softmax_cnn_knn_samples_dcgan_speech_model_clr0.0001_grl0.0001.ckpt-15000.npy'
softmax = np.load(name)

preds = softmax.argmax(axis=1)
acc = np.sum(preds == speaker_id) / float(len(softmax))


bins = set()
bins = bins | set(preds)
bins = np.array(sorted(list(bins)))

pred_counts = np.bincount(preds, minlength=n_classes)

d = 0.2
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.bar(np.arange(len(bins)), pred_counts[bins], align='center', alpha=0.7,
       label='Prediction acc. {0:.3f}'.format(acc))
ax.set_xticks(np.arange(len(bins)))
ax.set_xticklabels(bins)
ax.legend()
fig.savefig('pred_histogram_{}.png'.format(name))

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.imshow(softmax, aspect='auto', interpolation='nearest')
fig.savefig('softmaxes_{}.png'.format(name))
