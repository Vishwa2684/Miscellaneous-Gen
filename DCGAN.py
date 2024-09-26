import torch.nn as nn
random_noise = 128
class Generator(nn.Module):
    def __init__(self):
        """Output should be an image of 28x28x1 (WxHxC)"""
        super(Generator,self).__init__()
        # a fully connected network to project noise vector to 7*7*128
        self.fc = nn.Sequential(
            nn.Linear(random_noise,7*7*128),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(True)
        )
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,1,4,2,1),
            nn.Tanh()
        )

    def forward(self,z):
        x = self.fc(z)
        x = x.view(-1,128,7,7)
        return self.gen(x)
    


class Discriminator(nn.Module):
    """
        Input of the Discriminator is an image should be 1x28x28 (CxWxH)
        The output is a single neuron with sigmoid
        Downsampling an image 28x28x1 to 1
    """
    def __init__(self):
        super(Discriminator,self).__init__()
        """
            The output channels represent feature maps of an image in the following networks
        """
        self.model = nn.Sequential(
            #input of image 28x28x1 -> 14x14x64
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*256,1),
            nn.Sigmoid()
        )
    def forward(self,z):
        x = self.model(z)
        return self.fc(x)