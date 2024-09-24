import torch.nn as nn

# Set a seed

latent_dim = 100
class Generator(nn.Module):
    def block(self,ip_features,op_features,normalize=True):
            layers = [nn.Linear(ip_features,op_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(op_features,0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))    
            return layers
    def __init__(self):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            *self.block(latent_dim,128,False),
            *self.block(128,256),
            *self.block(256,512),
            *self.block(512,1024),
            nn.Linear(1024,1*28*28),
            nn.Tanh()
        )
    def forward(self,z):
            img = self.model(z)
            img = img.view(img.size(0),1,28,28)
            return img



class Discriminator(nn.Module):
    def block(self,ip,op):
        layers = [nn.Linear(ip,op),
            nn.LeakyReLU(0.2,inplace=True),
        ]
        return layers
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model =  nn.Sequential(
            *self.block(1*28*28,512),
            *self.block(512,256),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,z):
        img = z.view(z.size(0),-1)
        op =  self.model(img)
        return op   