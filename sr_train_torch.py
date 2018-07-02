from tqdm import tqdm
import torch
from data_processing import load_data, iterate_minibatches
from models import SpeakerRecognitionModel

# training params
SPEAKER_ID = None
SPEAKER_ID_OTHERS = None
LENGTH = 64
BATCH_SIZE = 256
VAL_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024
N_ITERS = int(1e5)

# model params
init_lr = 1e-4

# load data
data = load_data('data_16khz', '*.wav')
N_CLASSES = len(data['train'])
SPEAKER_ID_OTHERS = range(N_CLASSES)
N_TEST_RUNS = 100

data_training = iterate_minibatches(
    data['train'], SPEAKER_ID, SPEAKER_ID_OTHERS, BATCH_SIZE,
    shuffle=False, forever=True, length=LENGTH, one_hot_labels=False,
    apply_transform=False)

data_validation = iterate_minibatches(
    data['valid'], SPEAKER_ID, SPEAKER_ID_OTHERS, VAL_BATCH_SIZE,
    shuffle=False, forever=True, length=LENGTH, one_hot_labels=False)

data_testing = iterate_minibatches(
    data['test'], SPEAKER_ID, SPEAKER_ID_OTHERS, TEST_BATCH_SIZE,
    shuffle=False, forever=True, length=LENGTH, one_hot_labels=False)

model = SpeakerRecognitionModel(N_CLASSES).cuda()
optimizer = torch.optim.Adam(
    model.parameters(), lr=init_lr, betas=(0.5, 0.9), weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

model.train()

for i in tqdm(range(N_ITERS)):
    X, y = next(data_training)
    X, y = X.cuda(), y.cuda()
    X = X.unsqueeze(1)
    model.zero_grad()

    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    _, predicted = torch.max(y_hat, 1)
    accuracy = float((predicted == y).squeeze().sum()) / BATCH_SIZE
    loss.backward()
    optimizer.step()

    # learning rate schedule
    lr = init_lr * 0.9999 ** (i-10000)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if i % 100 == 0:
        print("Iteration {}, loss {}, accuracy {}, lr {}".format(
            i, float(loss), accuracy, lr))
        with torch.no_grad():
            X, y = next(data_validation)
            X, y = X.unsqueeze(1).cuda(), y.cuda()
            y_hat = model(X)

            _, predicted = torch.max(y_hat, 1)
            accuracy = float((predicted == y).squeeze().sum()) / VAL_BATCH_SIZE
            print("Validation accuracy {}".format(accuracy))

with torch.no_grad():
    accuracy = 0.0
    for _ in range(N_TEST_RUNS):
        X, y = next(data_testing)
        X, y = X.unsqueeze(1).cuda(), y.cuda()
        y_hat = model(X)

        _, predicted = torch.max(y_hat, 1)
        accuracy += float((predicted == y).squeeze().sum())
    print("Test accuracy {}".format(
        accuracy / (N_TEST_RUNS * TEST_BATCH_SIZE)))
