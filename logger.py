from tensorboardX import SummaryWriter
from plotting_utils import plot_spectrogram_to_numpy, reshape_to_matrix


class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_training(self, d_loss, g_loss, w_dist, d_fake_cost,
                     d_real_nspk_cost, d_real_spk_cost, d_gradient_penalty,
                     duration, iteration):
        self.add_scalar("D_loss", d_loss, iteration)
        self.add_scalar("G_loss", g_loss, iteration)
        self.add_scalar("W_dist", w_dist, iteration)
        self.add_scalar("d_fake_cost", d_fake_cost, iteration)
        self.add_scalar("d_real_nspk_cost", d_real_nspk_cost, iteration)
        self.add_scalar("d_real_spk_cost", -d_real_spk_cost, iteration)
        self.add_scalar("d_gradient_penalty", d_gradient_penalty, iteration)
        self.add_scalar("duration", duration, iteration)

    def log_validation(self, mel_real, mel_real_noisy, mel_fake, iteration):
        mel_real = mel_real.data[:32].cpu().numpy()
        mel_real_noisy = mel_real_noisy[:32].data.cpu().numpy()
        mel_fake = mel_fake.data[:32].cpu().numpy()

        mel_real = reshape_to_matrix(mel_real, 4, 8)
        mel_real_noisy = reshape_to_matrix(mel_real_noisy, 4, 8)
        mel_fake = reshape_to_matrix(mel_fake, 4, 8)

        self.add_image(
            "mel_real",
            plot_spectrogram_to_numpy(mel_real, 8, 6),
            iteration)
        self.add_image(
            "mel_real+noise",
            plot_spectrogram_to_numpy(mel_real_noisy, 8, 6),
            iteration)
        self.add_image(
            "mel_fake",
            plot_spectrogram_to_numpy(mel_fake, 8, 6),
            iteration)
