import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math
import itertools
import imageio
import natsort
from glob import glob
import torch.nn.utils.prune as prune
import random
import os
def get_data_loader(batch_size):
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.Resize((14)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))])

    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
    #idx = (train_dataset.targets==7)
    #train_dataset.targets = train_dataset.targets[idx]
    #train_dataset.data = train_dataset.data[idx]
    #print(train_dataset.data.size())
    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
def save_images(epoch, path, fixed_noise, num_test_samples, netG, device,use_fixed=False):
    z = torch.randn(num_test_samples, 4, 1, 1, device=device)
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None
  
    if use_fixed:
        generated_fake_images = netG(fixed_noise)
        path += 'fixed_noise/'
        title = 'Fixed Noise'
    else:
        generated_fake_images = netG(z)
        path += 'ideal/'
        title = 'Variable Noise'
    path += str(epoch)+'/'
    if not os.path.isdir(path):
        os.makedirs(path)
    for k in range(num_test_samples):
        f=open(path+'Noise_image_r_04_001_005.txt','a')
        np.savetxt(f,generated_fake_images[k].data.cpu().numpy().reshape(14,14))
        f.close()
def generate_images(epoch, path, fixed_noise, num_test_samples, netG, device, use_fixed=False):
    z = torch.randn(num_test_samples, 4, 1, 1, device=device)
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None
  
    if use_fixed:
        generated_fake_images = netG(fixed_noise)
        path += 'fixed_noise/'
        title = 'Fixed Noise'
    else:
        generated_fake_images = netG(z)
        path += 'variable_noise/'
        title = 'Variable Noise'
  
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6,6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(num_test_samples):
        i = k//4
        j = k%4
        ax[i,j].cla()
        ax[i,j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(14,14), cmap='Greys')
    label = 'Epoch_{}'.format(epoch+1)
    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path+label+'.png')

def save_gif(path, fps, fixed_noise=False):
    if fixed_noise==True:
        path += 'fixed_noise/'
    else:
        path += 'variable_noise/'
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    gif = []

    for image in images:
        gif.append(imageio.imread(image))
    imageio.mimsave(path+'animated.gif', gif, fps=fps)
import torch.nn as nn
import torch.nn.functional as F

class add_random_noise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, std):
        w_clip = torch.clip(w,-1*torch.max(torch.abs(w)),1*torch.max(torch.abs(w)))
        noise_tensor = torch.normal(0, std, size=w.size()).to(device)*torch.max(torch.abs(w))
        w_noise = w_clip+ noise_tensor
        return w_noise

    @staticmethod
    def backward(ctx, grad_input):
        return grad_input, None
_add_random_noise=add_random_noise.apply
class add_static_noise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, step):
        noise_tensor = torch.ones_like(w)*step
        w_noise = w+ noise_tensor
        return w_noise

    @staticmethod
    def backward(ctx, grad_input):
        return grad_input, None
_add_static_noise=add_static_noise.apply

class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
      super(Generator, self).__init__()
      self.network = nn.Sequential(
          #nz=4,ngf=2
          nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False),
          nn.BatchNorm2d(ngf*4),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(True),
  
          #nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
          #nn.BatchNorm2d(ngf),
          #nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
          nn.Tanh()
      )
  
    def forward(self, input):
      #add noise related to the max in one kernel
      #for index,param in enumerate(self.parameters()):
          #select the transposed conv layer

          #if  index == 0 or index == 3 or index == 6:
          #    s=param.data.size()
          #    for i in range(s[0]) :
          #        for j in range(s[1]):
          #            param.data[i][j]=_add_random_noise(param.data[i][j],0.005)
          #if  index == 0:              
              #if param.grad != None:
              #   print('gradient mean')
              #    print(torch.max(torch.abs(param.grad)))
              #print('weight mean')
              #print(torch.max(torch.abs(param.data)))
              #if param.grad != None:
                 #print(str(torch.max(torch.abs(param.grad)))+' '+str(torch.max(torch.abs(param.data))))
      output = self.network(input)
      return output

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
                
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                #nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ndf * 2),
                #nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
    def forward(self, input):
        #print(input.size())
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)
import argparse
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import copy
#from networks import Generator, Discriminator
#from utils import get_data_loader, generate_images, save_gif
# Set random seed for reproducibility.
seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

num_epochs=1000
ndf=32
ngf=32
nz=4
d_lr=0.001
g_lr=0.001
nc=1
batch_size=128
num_test_samples=16
#output_path='./results_large_all_r/'
output_path='./inference_all_0.05/'
fps=5
use_fixed=False
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser(description='DCGANS MNIST')
#    parser.add_argument('--num-epochs', type=int, default=100)
#    parser.add_argument('--ndf', type=int, default=32, help='Number of features to be used in Discriminator network')
#    parser.add_argument('--ngf', type=int, default=32, help='Number of features to be used in Generator network')
#    parser.add_argument('--nz', type=int, default=4, help='Size of the noise')
#    parser.add_argument('--d-lr', type=float, default=0.001, help='Learning rate for the discriminator')
#    parser.add_argument('--g-lr', type=float, default=0.001, help='Learning rate for the generator')
#    parser.add_argument('--nc', type=int, default=1, help='Number of input channels. Ex: for grayscale images: 1 and RGB images: 3 ')
#    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
#    parser.add_argument('--num-test-samples', type=int, default=16, help='Number of samples to visualize')
#    parser.add_argument('--output-path', type=str, default='./results/', help='Path to save the images')
#    parser.add_argument('--fps', type=int, default=5, help='frames-per-second value for the gif')
#    parser.add_argument('--use-fixed', action='store_true', help='Boolean to use fixed noise or not')

#    opt = parser.parse_args()
#    print(opt)

# Gather MNIST Dataset    
train_loader = get_data_loader(batch_size)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", device)

# Define Discriminator and Generator architectures
netG = Generator(nc, nz, ngf).to(device)
netD = Discriminator(nc, ndf).to(device)
#for name, module in netG.named_modules():
#    # prune 20% of connections in all 2D-conv layers 
#    if isinstance(module, torch.nn.ConvTranspose2d):
#        #prune.l1_unstructured(module, name='weight', amount=0.2)
#        prune.ln_structured(module, name="weight", amount=0.2, n=2, dim=1)
#        #print(module.weight)
# loss function
criterion = nn.BCELoss()

# optimizers
optimizerD = optim.Adam(netD.parameters(), lr=d_lr)
optimizerG = optim.Adam(netG.parameters(), lr=g_lr)

# initialize other variables
real_label = 1
fake_label = 0
num_batches = len(train_loader)
fixed_noise = torch.randn(num_test_samples, 4, 1, 1, device=device)
g_loss = []
d_loss = []
w_max = []
for epoch in range(num_epochs):
    print('epoch:'+str(epoch))
    for i, (real_images, _) in enumerate(train_loader):
        bs = real_images.shape[0]
        ##############################
        #   Training discriminator   #
        ##############################

        netD.zero_grad()
        real_images = real_images.to(device)
        label = torch.full((bs,), real_label, device=device)

        output = netD(real_images).to(device)
        label =label.type(torch.FloatTensor).to(device)
        lossD_real = criterion(output, label)
        lossD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(bs, nz, 1, 1, device=device)
        fake_images = netG(noise)
        label.fill_(fake_label)
        output = netD(fake_images.detach())
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        D_G_z1 = output.mean().item()
        lossD = lossD_real + lossD_fake
        optimizerD.step()

        ##########################
        #   Training generator   #
        ##########################

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake_images)
        #l2_lambda = 0.1
        #l2_reg = torch.tensor(0.)
        #for param in netG.parameters():
        #    l2_reg += torch.norm(param)
        
        lossG = criterion(output, label)


            #if grad1 != None:
                #print(index)
                #print(grad)
        #with torch.no_grad():
       

        #print(lossG)
        #lossG += l2_lambda * l2_reg
        #print(lossG)
        #optimizerG.zero_grad()
        lossG.backward(retain_graph=True)
        model2 = copy.deepcopy(netG)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=g_lr)
        
        for paramName, paramValue, in netG.named_parameters():
            for netCopyName, netCopyValue, in model2.named_parameters():
                if paramName == netCopyName:
                    if paramValue.grad != None:
                        netCopyValue.grad = paramValue.grad.clone()        

        
        for index, param in enumerate(model2.parameters()):
            if param.grad != None:
                #print(torch.sign(param.grad)*torch.max(param.data))
                #print(param.data)
                param.data +=0.01*torch.sign(param.grad)*torch.max(param.data)
                #print(param.data)
        fake_images2 = model2(noise)
        output2 = netD(fake_images2)
        #print(output2)
        lossG2 = criterion(output2, label)
        model2.zero_grad()
        lossG2.backward()
        #optimizer2.step()
        #
        for index, param in enumerate(netG.parameters()):
            grad1 = param.grad
            for index2, param2 in enumerate(model2.parameters()):
                 
                 if index2 == index:
                     #print(index2)
                     grad2 = param2.grad
                     #print(grad1)
                     #print(grad2)
                     if grad1 != None and grad2 != None:
                         #print((grad1-grad2).norm())
                         lossG += .4*(grad1-grad2).norm()
        lossG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i==48:
            print('Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, Discriminator - D(G(x)): {:.2f}, Generator - D(G(x)): {:.2f}'.format(epoch+1, num_epochs, 
                                                        i+1, num_batches, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
            g_loss.append( lossG.item())
            d_loss.append( lossD.item())
            for index,param in enumerate(netG.parameters()):
              if  index == 0 and param.grad != None:              
                 print('Gradient and weight for layer 1:' + str(torch.max(torch.abs(param.grad)))+' '+str(torch.max(torch.abs(param.data))))
                 w_max.append(torch.max(torch.abs(param.data)))
    netG.eval()
    num_test_samples=10
    if epoch in [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,999]:
        print(epoch)
        for i in range(200):
            netG.load_state_dict(torch.load('model/model_final_large_all_r=0.4_0.01.pth')['generator'],strict=False)
            for index,param in enumerate(netG.parameters()):
                if index == 0 or index == 3 or index == 6:
                    s=param.data.size()
                    for i in range(s[0]) :
                        for j in range(s[1]):
                            param.data[i][j]=_add_random_noise(param.data[i][j],0.05)
            save_images(epoch, output_path, fixed_noise, num_test_samples, netG, device,use_fixed=use_fixed)
    #generate_images(epoch, output_path, fixed_noise, num_test_samples, netG, device, use_fixed=use_fixed)
    netG.train()
    
    torch.save({
                'generator' : netG.state_dict(),
                'discriminator' : netD.state_dict(),
                'optimizerG' : optimizerG.state_dict(),
                'optimizerD' : optimizerD.state_dict()
                }, 'model/model_final_large_all_r=0.4_0.01.pth')
    #if epoch%100 == 0:
    #    step_plot=[]  
    #    lossGcopy_plot=[]
    #    for step in torch.arange(-0.01,0.011,0.001):
    #        #print(step)
    #        netGcopy = copy.deepcopy(netG)
    #        for index,param in enumerate(netGcopy.parameters()):
    #            if index == 0 or index == 3 or index == 6:
    #                s = param.size()
    #                for i in range(s[0]) :
    #                    for j in range(s[1]):
    #                        param.data[i][j]=_add_static_noise(param.data[i][j],step)
    #        for i, (real_images, _) in enumerate(train_loader):
    #            fake_images_copy = netGcopy(noise)
    #            label.fill_(real_label)
    #            outputcopy = netD(fake_images_copy)        
    #            lossGcopy = criterion(outputcopy, label)
    #            if i == 48:
    #                step_plot.append(step)
    #                lossGcopy_plot.append(lossGcopy.item())
            
    #    plt.figure()
    #    plt.plot(step_plot,lossGcopy_plot)

    
#plt.figure()
#plt.plot(g_loss)
#plt.figure()
#plt.plot(d_loss)
#plt.figure()
#plt.plot(w_max)
# Save gif:
#save_gif(output_path, fps, fixed_noise=use_fixed)




