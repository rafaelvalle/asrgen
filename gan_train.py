""" adapted from https://github.com/martinarjovsky/WassersteinGAN """
import time

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from data_processing import load_data, iterate_minibatches
from logger import Logger
from models import Generator, Discriminator
from utils import weights_init_discriminator, weights_init_generator
from utils import load_checkpoint, save_checkpoint

torch.manual_seed(1)

# ======================== PARAMS ==========================
SPEAKER_ID = 101
SPEAKER_ID_OTHERS = [i for i in range(101) if i != SPEAKER_ID]
BETA = 0 if (SPEAKER_ID is None or not len(SPEAKER_ID_OTHERS)) else 1.0
REG_NOISE = 1e-8
DATA_FOLDER = 'data_16khz'
USE_TRAIN = True
MODEL_PATH = None
PLOT_FREQ = 100
SAVE_FREQ = 500
BATCH_SIZE = 256
BEGIN_ITERS = 0
END_ITERS = 50001
N_CHANNELS = 1
N_MEL_CHANNELS = 64
MEL_NORM = None
MEL_HTK = True
LENGTH = 64
DIM = 64
GLR = 1e-4
DLR = 1e-4

OUTPUT_DIRECTORY = '/tts/runs/asrgen/speaker{}/relgan_orthogonal_norm{}_htk{}_train{}_regnoiz{}_bs{}'.format(
    SPEAKER_ID, MEL_NORM, MEL_HTK, USE_TRAIN, REG_NOISE,
    BATCH_SIZE)
NAME = 'dcgan_speaker{}_beta{}_clr{}_grl{}'.format(
    SPEAKER_ID, str(BETA), str(DLR), str(GLR))

G_net = Generator(DIM, N_MEL_CHANNELS, LENGTH).cuda()
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

all_data = load_data(DATA_FOLDER, '*.wav', N_MEL_CHANNELS, MEL_NORM, MEL_HTK)
if USE_TRAIN:
    data = all_data['train']
else:
    data = [[torch.cat((all_data['validation'][i][0], all_data['test'][i][0]), dim=1), all_data['validation'][i][1]]
            for i in range(len(all_data['validation']))]

train_generator = iterate_minibatches(
    data, SPEAKER_ID, SPEAKER_ID_OTHERS, BATCH_SIZE,
    length=LENGTH, shuffle=False)

logger = Logger(OUTPUT_DIRECTORY)


for iteration in range(BEGIN_ITERS, END_ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in D_net.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in G_net update

    reg_noise = Variable(torch.FloatTensor(
        BATCH_SIZE, N_MEL_CHANNELS, LENGTH).normal_(0.0, REG_NOISE)).cuda()

    # real data from speaker, add regularization noise
    data, labels = next(train_generator)
    real_data_spk = data[:BATCH_SIZE]
    real_data_spk = Variable(real_data_spk).cuda()
    real_data_spk += reg_noise

    D_net.zero_grad()
    # train with real from target speaker and fake data
    D_real_spk = D_net(real_data_spk)

    # train with fake data
    noise = Variable(torch.randn(BATCH_SIZE, 128), volatile=True).cuda()
    fake_data = Variable(G_net(noise).data)
    D_fake = D_net(fake_data)
    D_real_fake_loss = -torch.log(F.sigmoid(D_real_spk - D_fake))
    D_real_fake_loss = D_real_fake_loss.mean()
    D_real_fake_loss.backward()

    D_real_realnspk_loss = 0.0
    if BETA:
        data, labels = next(train_generator)
        real_data_spk = data[:BATCH_SIZE]
        real_data_spk = Variable(real_data_spk).cuda()
        real_data_spk += reg_noise

        # train with real from target speaker and fake data
        D_real_spk = D_net(real_data_spk)

        # real data not from speaker
        real_data_nspk = data[BATCH_SIZE:]
        real_data_nspk = Variable(real_data_nspk).cuda()
        # train with real from other speakers
        D_real_nspk = D_net(real_data_nspk)
        D_real_realnspk_loss = -torch.log(F.sigmoid(D_real_spk - D_real_nspk))
        D_real_realnspk_loss = D_real_realnspk_loss.mean()
        D_real_realnspk_loss.backward()

        data, labels = next(train_generator)
        # real data not from speaker
        real_data_nspk = data[BATCH_SIZE:]
        real_data_nspk = Variable(real_data_nspk).cuda()
        D_real_nspk = D_net(real_data_nspk)

        # fake data
        noise = Variable(torch.randn(BATCH_SIZE, 128), volatile=True).cuda()
        fake_data = Variable(G_net(noise).data)
        D_fake = D_net(fake_data)

        D_fake_realnspk_loss = -torch.log(F.sigmoid(D_fake - D_real_nspk))
        D_fake_realnspk_loss = D_fake_realnspk_loss.mean()
        D_fake_realnspk_loss.backward()

    D_cost = D_real_fake_loss + D_real_realnspk_loss + D_fake_realnspk_loss
    D_optimizer.step()

    ############################
    # (2) Update G network
    ###########################
    for p in D_net.parameters():
        p.requires_grad = False  # to avoid computation
    G_net.zero_grad()

    noise = Variable(torch.randn(BATCH_SIZE, 128)).cuda()
    fake = G_net(noise)
    D_fake = D_net(fake)

    data, labels = next(train_generator)
    real_data_spk = data[:BATCH_SIZE]
    real_data_spk = Variable(real_data_spk).cuda()
    real_data_spk += reg_noise
    D_real_spk = D_net(real_data_spk)

    G_fake_real_loss = -torch.log(F.sigmoid(D_fake - D_real_spk))
    G_fake_real_loss =  G_fake_real_loss.mean()
    G_cost = G_fake_real_loss
    G_fake_real_loss.backward()
    G_optimizer.step()

    duration = time.time() - start_time
    print("iteration {}, duration {} d_cost {} g_cost {}".format(
        iteration, duration, D_cost, G_cost))
    logger.log_training(D_cost, G_cost, 0.0, D_real_fake_loss, D_real_realnspk_loss,
                        0.0, 0.0, duration, iteration)

    if iteration % PLOT_FREQ == 0:
        reg_noise = Variable(torch.FloatTensor(
            BATCH_SIZE, N_MEL_CHANNELS, LENGTH).normal_(0.0, REG_NOISE)).cuda()
        logger.log_validation(real_data_spk, real_data_spk+reg_noise,
                              fake_data, iteration)

    if iteration % SAVE_FREQ == 0:
        checkpoint_path = "{}/checkpoint_{}_{}".format(
            OUTPUT_DIRECTORY, NAME, iteration)
        save_checkpoint(D_net, G_net, D_optimizer, G_optimizer, iteration,
                        checkpoint_path)
