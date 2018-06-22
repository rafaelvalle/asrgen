import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, DIM, bn=False):
        super(Generator, self).__init__()
        self.DIM = DIM
        self.preprocess = nn.Linear(128, 4*4*8*DIM)
        self.bn = nn.BatchNorm2d(8*DIM)
        self.relu = nn.ReLU(True)

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(8*DIM, 4*DIM, 4, 2, 1),
            nn.BatchNorm2d(4*DIM),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 4, 2, 1),
            nn.BatchNorm2d(2*DIM),
            nn.ReLU(True),
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 4, 2, 1),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        self.deconv_out = nn.ConvTranspose2d(DIM, 1, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 8*self.DIM, 4, 4)
        output = self.bn(output)
        output = self.relu(output)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, self.DIM, self.DIM)


class Discriminator(nn.Module):
    def __init__(self, DIM):
        super(Discriminator, self).__init__()
        self.DIM = DIM
        self.main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.BatchNorm2d(2*DIM),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.BatchNorm2d(4*DIM),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4*DIM, 8*DIM, 5, stride=2, padding=2),
            nn.BatchNorm2d(8*DIM),
            nn.LeakyReLU(0.2)
        )

        self.output = nn.Linear(4*4*8*DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, self.DIM, self.DIM)
        out = self.main(input)
        out = out.view(-1, 4*4*8*self.DIM)
        out = self.output(out)
        return out.view(-1)
