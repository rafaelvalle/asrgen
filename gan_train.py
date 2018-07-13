""" adapted from https://github.com/martinarjovsky/WassersteinGAN """
import time

import torch
import torch.autograd as autograd
import torch.optim as optim

from data_processing import load_data, iterate_minibatches
from logger import Logger
from models import Generator, Discriminator
from utils import weights_init_discriminator, weights_init_generator
from utils import calc_gradient_penalty
from utils import load_checkpoint, save_checkpoint

torch.manual_seed(1)

# ======================== PARAMS ==========================
SPEAKER_ID = 101
SPEAKER_ID_OTHERS = range(101)
CRITIC_ITERS = 10
BETA = 0 if (SPEAKER_ID is None or not len(SPEAKER_ID_OTHERS)) else 1.0
REG_NOISE = 1e-5
DATA_FOLDER = 'data_16khz'
USE_TRAIN = False
MODEL_PATH = None
PLOT_FREQ = 100
SAVE_FREQ = 500
BATCH_SIZE = 64
BEGIN_ITERS = 0
END_ITERS = 30001
LAMBDA = 10  # Gradient penalty lambda hyperparameter
N_CHANNELS = 1
DIM = 64
LENGTH = 64
GLR = 1e-4
DLR = 1e-4

OUTPUT_DIRECTORY = '/tts/runs/asrgen/speaker{}/run_0_val_usetrain{}_citers{}_regnoise{}'.format(
    SPEAKER_ID, USE_TRAIN, CRITIC_ITERS, REG_NOISE)
NAME = 'dcgan_speaker{}_beta{}_clr{}_grl{}'.format(
    SPEAKER_ID, str(BETA), str(DLR), str(GLR))

G_net = Generator(DIM).cuda()
D_net = Discriminator(DIM).cuda()
G_net.apply(weights_init_generator)
D_net.apply(weights_init_discriminator)
print(G_net)
print(D_net)

D_optimizer = optim.Adam(D_net.parameters(), lr=DLR, betas=(0.5, 0.9))
G_optimizer = optim.Adam(G_net.parameters(), lr=GLR, betas=(0.5, 0.9))

if MODEL_PATH is not None:
    D_net, G_net, D_optimizer, G_optimizer = load_checkpoint(
        MODEL_PATH, D_net, G_net, D_optimizer, G_optimizer)

one = torch.FloatTensor([1]).cuda()
mone = one * -1

SAMPLE_SIZE = BATCH_SIZE * 2 if BETA else BATCH_SIZE
all_data = load_data(DATA_FOLDER, '*.wav')
if USE_TRAIN:
    data = all_data['train']
else:
    data = [[torch.cat((all_data['validation'][i][0], all_data['test'][i][0]), dim=1), all_data['validation'][i][1]]
            for i in range(len(all_data['validation']))]

train_generator = iterate_minibatches(
    data, SPEAKER_ID, SPEAKER_ID_OTHERS, SAMPLE_SIZE,
    length=LENGTH, shuffle=False)

logger = Logger(OUTPUT_DIRECTORY)


for iteration in range(BEGIN_ITERS, END_ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in D_net.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in G_net update

    for iter_d in range(CRITIC_ITERS):
        reg_noise = torch.autograd.Variable(torch.FloatTensor(
            BATCH_SIZE, DIM, DIM).normal_(0.0, REG_NOISE)).cuda()

        # real data from speaker, add regularization noise
        data, labels = next(train_generator)
        real_data_spk = data[:BATCH_SIZE]
        real_data_spk = torch.autograd.Variable(real_data_spk).cuda()
        real_data_spk += reg_noise

        D_net.zero_grad()
        # train with real from target speaker
        D_real_spk = D_net(real_data_spk).mean()
        D_real_spk.backward(mone)

        D_real_nspk = 0.0
        if BETA:
            # real data not from speaker
            real_data_nspk = data[BATCH_SIZE:]
            real_data_nspk = torch.autograd.Variable(real_data_nspk).cuda()
            # train with real from other speakers
            D_real_nspk = BETA * D_net(real_data_nspk).mean()
            D_real_nspk.backward(one)

        # train with fake data
        noise = autograd.Variable(
            torch.randn(BATCH_SIZE, 128), volatile=True).cuda()
        fake_data = autograd.Variable(G_net(noise).data)
        D_fake = D_net(fake_data).mean()
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(
            D_net, real_data_spk.data, fake_data.data, BATCH_SIZE, LAMBDA)
        gradient_penalty.backward()

        D_cost = D_fake + D_real_nspk - D_real_spk + gradient_penalty
        Wasserstein_D = D_real_spk - D_fake
        D_optimizer.step()

    ############################
    # (2) Update G network
    ###########################
    for p in D_net.parameters():
        p.requires_grad = False  # to avoid computation
    G_net.zero_grad()

    noise = autograd.Variable(torch.randn(BATCH_SIZE, 128)).cuda()
    fake = G_net(noise)
    G = D_net(fake).mean()
    G.backward(mone)
    G_cost = -G
    G_optimizer.step()

    duration = time.time() - start_time
    print("iteration {}, duration {} d_cost {} g_cost {}".format(
        iteration, duration, D_cost, G_cost))
    logger.log_training(D_cost, G_cost, Wasserstein_D, D_fake,
                        D_real_nspk, D_real_spk, gradient_penalty,
                        duration, iteration)

    if iteration % PLOT_FREQ == 0:
        reg_noise = torch.autograd.Variable(
            torch.FloatTensor(DIM, DIM).normal_(0.0, REG_NOISE)).cuda()
        logger.log_validation(real_data_spk, real_data_spk+reg_noise,
                              fake_data, iteration)

    if iteration % SAVE_FREQ == 0:
        checkpoint_path = "{}/checkpoint_{}_{}".format(
            OUTPUT_DIRECTORY, NAME, iteration)
        save_checkpoint(D_net, G_net, D_optimizer, G_optimizer, iteration,
                        checkpoint_path)
