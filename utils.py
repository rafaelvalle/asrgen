import os
import numpy as np
from scipy.io.wavfile import read
import torch


def load_checkpoint(checkpoint_path, D_net, G_net, D_optimizer, G_optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    D_optimizer.load_state_dict(checkpoint_dict['D_optimizer'])
    G_optimizer.load_state_dict(checkpoint_dict['G_optimizer'])
    D_net.load_state_dict(checkpoint_dict['D_net'])
    G_net.load_state_dict(checkpoint_dict['G_net'])
    print("Loaded checkpoint '{}' (iteration {})" .format(
        checkpoint_path, 0))
    return D_net, G_net, D_optimizer, G_optimizer


def save_checkpoint(D_net, G_net, D_optimizer, G_optimizer, iteration,
                    filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'D_net': D_net.state_dict(),
                'G_net': G_net.state_dict(),
                'D_optimizer': D_optimizer.state_dict(),
                'G_optimizer': G_optimizer.state_dict(),
                'iteration': iteration}, filepath)


def load_wav_to_torch(full_path, sr):
    sampling_rate, data = read(full_path)
    assert sr == sampling_rate, "{} SR doesn't match {} on path {}".format(
        sr, sampling_rate, full_path)
    return torch.FloatTensor(data.astype(np.float32))


def weights_init_discriminator(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #torch.nn.init.xavier_normal(
        #    m.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.normal(m.weight, 0.0, 0.05)
    elif classname.find('Linear') != -1:
        #torch.nn.init.xavier_normal(
        #    m.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.normal(m.weight, 0.0, 0.05)


def weights_init_generator(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #torch.nn.init.xavier_normal(
        #    m.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.normal(m.weight, 0.0, 0.05)
    elif classname.find('Linear') != -1:
        #torch.nn.init.xavier_normal(
        #    m.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.normal(m.weight, 0.0, 0.05)
    else:
        print(classname)


def calc_gradient_penalty(D_net, real_data, fake_data,  batch_size, lamb):
    alpha = torch.rand(batch_size, 1, 1).expand(real_data.size()).cuda()
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = D_net(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    return gradient_penalty