import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np

softmax_wgan = np.load('softmax_cnn_knn_samples_dcgan_speech_model_clr0.0001_grl0.0001.ckpt-15000.npy')
softmax_mixed = np.load('softmax_cnn_knn_samples_dcgan_speech_model_sid0_clr0.0001_grl0.0001_beta1.0.ckpt-4000.npy')
speaker_id = 10
n_classes = 103
print("Comparing {} and {}".format(softmax_wgan.shape, softmax_mixed.shape))

preds_wgan = softmax_wgan.argmax(axis=1)
preds_mixed = softmax_mixed.argmax(axis=1)
acc_wgan = np.sum(preds_wgan == speaker_id) / float(len(softmax_wgan))
acc_mixed = np.sum(preds_mixed == speaker_id) / float(len(softmax_mixed))

bins = set()
bins = bins | set(preds_mixed)
bins = bins | set(preds_wgan)
bins = np.array(sorted(list(bins)))

pred_counts_mixed = np.bincount(preds_mixed, minlength=n_classes)
pred_counts_wgan = np.bincount(preds_wgan, minlength=n_classes)

print("Generating plots")
d = 0.2
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.bar(np.arange(len(bins)), pred_counts_mixed[bins], align='center',
       alpha=0.7, label='Mixed loss, err. rate {0:.3f}'.format(1-acc_mixed))
ax.bar(np.arange(len(bins)), pred_counts_wgan[bins], align='center',
       alpha=0.7, label='IWGAN loss, err. rate {0:.3f}'.format(1-acc_wgan))
ax.set_xticks(np.arange(len(bins)))
ax.set_xticklabels(bins)
ax.legend()
fig.savefig('pred_histogram_sid{}_nclasses{}.png'.format(speaker_id, n_classes))
