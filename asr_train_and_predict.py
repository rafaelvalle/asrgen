import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from data_processing import *
from utils import plot_confusion_matrix, plot_histogram
import cPickle as pkl

speech_data = load_data(
    '/media/steampunkhd/rafaelvalle/datasets/AUDIO/speech_data/NIST_2004/',
    '*64.npy')

spk_id = None # train with balanced dataset
n_classes = 102
batch_size = 256
speech_training = iterate_minibatches(
    speech_data['train'], lbl_id=spk_id, batch_size=batch_size,
    shuffle=False, forever=True, length=64)

speech_validation = iterate_minibatches(
    speech_data['valid'], lbl_id=spk_id, batch_size=256,
    shuffle=True, forever=True, length=64)

speech_testing = iterate_minibatches(
    speech_data['test'], lbl_id=spk_id, batch_size=256,
    shuffle=True, forever=True, length=64)


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 4096])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                          strides=[1, 4, 4, 1], padding='SAME')
#
# start building the conv layer
# ----------------------------------------------------------------------------
x_image = tf.reshape(x, [-1,64,64,1])

# first conv layer
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# h_pool1 = max_pool_4x4(h_conv1)


# second conv layer
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# h_pool2 = max_pool_4x4(h_conv2)

# fc layer
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
W_fc1 = weight_variable([16 * 16 * 64, 1010])
# b_fc1 = bias_variable([1024])
b_fc1 = bias_variable([1010])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
# W_fc2 = weight_variable([1024, 10])
W_fc2 = weight_variable([1010, n_classes])
# b_fc2 = bias_variable([10])
b_fc2 = bias_variable([n_classes])

# output of last layer, softmax and prediction fn for CNN
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
softmax_fn = tf.nn.softmax(y_conv)
prediction_fn = tf.argmax(y_conv, 1)

# final layer
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#
# train
# ---------------------------------------------------------------------------------------
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.global_variables_initializer())

init_op = tf.global_variables_initializer()
# save model
saver = tf.train.Saver()

# train model
sess.run(init_op)
TRAIN = False
if TRAIN:
    for i in range(10100):
        X, y = next(speech_training)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: np.reshape(X, [-1, 4096]), y_: y, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: np.reshape(X, [-1, 4096]),
                                  y_: y, keep_prob: 0.5})

        if i % 1000 == 0:
            X, y = next(speech_validation)
            val_accuracy = accuracy.eval(feed_dict={
                x: np.reshape(X, [-1, 4096]), y_: y, keep_prob: 1.0})
            print("step %d, validation accuracy %g" % (i, val_accuracy))

        if i % 5000 == 0:
            save_path = saver.save(sess, "./saver/fullmodel" + str(i) + ".ckpt")
            print("Model saved in file: %s" % save_path)
saver.restore(sess, './saver/fullmodel10000.ckpt')

# embbed real data by running it through up to penultimate layer of classifier
samples = []
embeddings = []
labels = []
n_batches = 200
print("Sampling {} embedded examples from real data".format(
    n_batches*batch_size))

for i in range(n_batches):
    X, y = next(speech_training)
    output = h_fc1_drop.eval(feed_dict={
        x: np.reshape(X, [-1, 4096]), keep_prob: 1.0})
    samples.extend(X)
    embeddings.extend(output)
    labels.extend(y)

samples = np.array(samples)
embeddings = np.array(embeddings)
labels = np.array(labels)
labels = np.argmax(labels, axis=1)
bincount = np.bincount(labels)
print("Real data sample shape {}".format(samples.shape))
print("Label histogram {}".format(bincount))

# train knn classifier with embedded real data
KNN = False
if KNN:
    K = 5
    knn_clf = KNeighborsClassifier(n_neighbors=K)
    print("Fitting KNN Classifier")
    knn_clf.fit(embeddings, labels)
    # save KNN classifier
    with open('knn_classifier_{}.pkl'.format(n_classes), 'wb') as f:
        pkl.dump(knn_clf, f)
with open('knn_classifier_{}.pkl'.format(n_classes), 'rb') as f:
        knn_clf = pkl.load(f)

#
# use embedded real data for sanity tests, THIS TAKES > 20mins
#
SANITY_TESTS = False
if SANITY_TESTS:
    print("Predicting real data using KNN classifier")
    knn_preds = knn_clf.predict(embeddings)

    # avoid out of memory by looping
    preds = []
    stride = 100
    for i in np.arange(0, len(samples), stride):
        preds.extend(prediction_fn.eval(feed_dict={
            x: np.reshape(samples[i:i+stride], [-1, 4096]), keep_prob: 1.0}))
    preds = np.array(preds)

    print("Saving Confusion Matrices for KNN based predictions")
    cm = confusion_matrix(
        y_true=labels, y_pred=preds, labels=np.arange(0, n_classes))
    plot_confusion_matrix(cm, np.arange(0, n_classes), normalize=True,
                        filename='conf_mat_cnn.png')
    cm = confusion_matrix(
        y_true=labels, y_pred=knn_preds, labels=np.arange(0, n_classes))
    plot_confusion_matrix(cm, np.arange(0, n_classes), normalize=True,
                        filename='conf_mat_knn.png')

#
# embed fake data by running it through up to penultimate layer of classifier
#
fake_batch = np.load('dcgan_samples_64_sid_0_5000.npy')
output = h_fc1_drop.eval(feed_dict={
    x: np.reshape(fake_batch, [-1, 4096]), keep_prob: 1.0})
print("Predicting fake data using KNN classifier")
knn_preds = knn_clf.predict(output)
preds = prediction_fn.eval(feed_dict={
    x: np.reshape(fake_batch, [-1, 4096]), keep_prob: 1.0})
softmax = softmax_fn.eval(feed_dict={
    x: np.reshape(fake_batch, [-1, 4096]), keep_prob: 1.0})

# test on different data
name = 'wgan_sid_0_5000'
print("Computing histogram of predicted probabilities")
plot_histogram(softmax.flatten(),
               filename='histogram_probs_{}.png'.format(name))

print("Computing histogram of highest probabilities")
plot_histogram(softmax[np.arange(len(softmax)), np.argmax(softmax, axis=1)],
               filename='histogram_max_probs_{}.png'.format(name))

print("Computing and saving Confusion Matrix")
cm = confusion_matrix(y_true=preds, y_pred=knn_preds,
                      labels=np.arange(0, n_classes))
plot_confusion_matrix(cm, np.arange(0, n_classes), normalize=True,
                      filename='conf_mat_cnn_knn_{}.png'.format(name))
np.save('conf_mat_cnn_knn_{}.npy'.format(name), cm)
np.save('softmax_cnn_knn_{}.npy'.format(name), softmax)