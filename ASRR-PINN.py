import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0")#"cuda:0" if torch.cuda.is_available() else
import numpy as np
from numpy import genfromtxt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import time
import gc
import math
from scipy.stats import qmc
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import csv
writer = SummaryWriter(log_dir='runs/grid1')
seed = 1
#dimension of x, y and z
a = 0.015
b = 0.015
c = 0.001

#Truncation Factor of x, y and z
P=25
Q=25
S=25

#initial segmentation
x_grid = 20
y_grid = 20
z_grid = 20

#thermal properties
T_std = 293.15
T0=293.15
k1 = 0.0005
k2=-0.4571
b2=282.4
qnum = 20

#hyper-para of ASRR algorithm
adap_begin = 4000
iterations = 4925
adap_end = 100
discrete_times = 3

#1:ASRR | 2:RAR | 3:RAD | 4:Uniform | 5:Halton | 6:LHS | 7:SOBOL | 8:Grid
resample_mode = 4

#MLP
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.eigenx = torch.linspace(0, P-1, steps=P).to(device)
        self.eigeny = torch.linspace(0, Q-1, steps=Q).to(device)
        self.eigenz = torch.linspace(0, S-1, steps=S).to(device)
        
        self.lin1 = nn.Linear(2,1024)
        self.lin2 = nn.Linear(1024,1024)
        self.lin3 = nn.Linear(1024,2048)
        self.lin4 = nn.Linear(2048,2048)
        self.lin5 = nn.Linear(2048,4096)
        self.lin6 = nn.Linear(4096,4096)
        self.lin7 = nn.Linear(4096,P*Q*S)
        self.acti1 = nn.LeakyReLU()
        self.acti2 = nn.LeakyReLU()
        self.acti3 = nn.LeakyReLU()
        self.acti4 = nn.LeakyReLU()
        self.acti5 = nn.LeakyReLU()
        self.acti6 = nn.LeakyReLU()
        
    def forward(self, x,y,z,b1,k):
        weight = torch.tensor((b1,k)).to(torch.float32)
        weight = torch.reshape(weight,((1,2))).to(device)
        weight = self.lin1(weight)
        weight = self.acti1(weight)
        weight = self.lin2(weight)
        weight = self.acti2(weight)
        weight = self.lin3(weight)
        weight = self.acti3(weight)
        weight = self.lin4(weight)
        weight = self.acti4(weight)
        weight = self.lin5(weight)
        weight = self.acti5(weight)
        weight = self.lin6(weight)
        weight = self.acti6(weight)
        weight = self.lin7(weight)
        weight = torch.reshape(weight,((P,Q,S)))
        X=torch.cos(self.eigenx*np.pi/a*x)[:,:,None,None]
        Y=torch.cos(self.eigeny*np.pi/b*y)[:,None,:,None]
        Z = torch.cos((self.eigenz*np.pi+0.5*np.pi)*z/c)[:,None,None,:]
        F = torch.einsum('ijkl,ijlm,ijmn->ijln', X, Y, Z)  # shape: (10000, 30, 30, 30)
        return torch.einsum('bijk,ijk->b',F,weight)[:,None]+(T0)/T_std

#import power and floorplan
Power=genfromtxt('facesim.ptrace')
FUnit=genfromtxt('ev6.flp')
num_power = Power.shape[1]
#calculate power density of coordinate (x,y,z)
def f(x,y,z):
    if(z>=0.0004 and z<=0.0007):
        for i in range(num_power):
            if(x>=FUnit[i,3] and x<=FUnit[i,3]+FUnit[i,1] and y>=FUnit[i,4] and y<=FUnit[i,4]+FUnit[i,2]):
               return Power[1,i]*qnum/(FUnit[i,1]*FUnit[i,2]*0.0003)
        return 0
    else:
        return 0

net = Net()
net1 = net.to(device)

if resample_mode == 1:
    num_p = x_grid*y_grid*z_grid
elif resample_mode == 2 or resample_mode == 3:
    num_p = 27000
else:
    num_p = 32050

#pde loss for other sampling methods
def pde(x,y,z,p,net,b1,k):

    u = net(x,y,z,b1,k) # the dependent variable u is given by the network based on independent variables x,t
    ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
    u_x = torch.autograd.grad(outputs=u,inputs=x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0:1]
    u_x_x= torch.autograd.grad(outputs=u_x,inputs=x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0:1]
    
    u_y = torch.autograd.grad(outputs=u,inputs=y, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0:1]
    u_y_y= torch.autograd.grad(outputs=u_y,inputs=y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:,0:1]
    
    u_z = torch.autograd.grad(outputs=u,inputs=z, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0:1]
    u_z_z= torch.autograd.grad(outputs=u_z,inputs=z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:,0:1]
    
    r = 130*k*u_x_x*T_std + 130*k*u_y_y*T_std + 130*k*u_z_z*T_std + p * (b1 + k1 * u * T_std)
    allocated = torch.cuda.memory_allocated()
    ans = torch.mul(r,r)
    
    return ans/10000000000

#pde loss for ASRR
def pde1(x,y,z,p,net,b1,k,epoch):

    u = net(x,y,z,b1,k) # the dependent variable u is given by the network based on independent variables x,t
    ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
    
    u_x = torch.autograd.grad(outputs=u,inputs=x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0:1]
    u_x_x= torch.autograd.grad(outputs=u_x,inputs=x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0:1]
    
    u_y = torch.autograd.grad(outputs=u,inputs=y, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0:1]
    u_y_y= torch.autograd.grad(outputs=u_y,inputs=y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:,0:1]
    
    u_z = torch.autograd.grad(outputs=u,inputs=z, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0:1]
    u_z_z= torch.autograd.grad(outputs=u_z,inputs=z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:,0:1]
     
    r = 130*k*u_x_x*T_std + 130*k*u_y_y*T_std + 130*k*u_z_z*T_std + p * (0.853425 + k1 * u * T_std)
    allocated = torch.cuda.memory_allocated()
    if(epoch >= adap_begin):
        r_x = 130 * torch.autograd.grad(outputs=u_x_x, inputs=x, grad_outputs=torch.ones_like(u_x_x), create_graph=True)[0][:,0:1] * T_std + p * k1 * u_x * T_std
        r_y = 130 * torch.autograd.grad(outputs=u_y_y, inputs=y, grad_outputs=torch.ones_like(u_y_y), create_graph=True)[0][:,0:1] * T_std + p * k1 * u_y * T_std
        r_z = 130 * torch.autograd.grad(outputs=u_z_z, inputs=z, grad_outputs=torch.ones_like(u_z_z), create_graph=True)[0][:,0:1] * T_std + p * k1 * u_z * T_std
        h1 = r_x**2+r_y**2+r_z**2
    else:
        h1 = r
    ans = torch.mul(r,r)
    
    return ans/10000000000, h1



decay_interval = 550
decay_rate = 0.75

all_zeros = np.zeros((num_p,1))

optimizer = torch.optim.Adam(net1.parameters(),lr=0.0001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

x_add = np.arange(0, a, a/x_grid)
y_add = np.arange(0, b, b/y_grid)
z_add = np.arange(0, c, c/z_grid)
ms_x_add, ms_y_add, ms_z_add  = np.meshgrid(x_add, y_add, z_add)
## Just because meshgrid is used, we need to do the following adjustment
x_add = np.ravel(ms_x_add).reshape(-1,1)
y_add = np.ravel(ms_y_add).reshape(-1,1)
z_add = np.ravel(ms_z_add).reshape(-1,1)


matrix_W = torch.ones((num_p,1)).float().to(device)
matrix_Size = np.ones((num_p,1))


if(resample_mode==4):
    x_input = np.random.uniform(low=0, high=a, size=(num_p,1))
    y_input = np.random.uniform(low=0, high=b, size=(num_p,1))
    z_input = np.random.uniform(low=0, high=c, size=(num_p,1))
    p_input = np.zeros((num_p,1))
    for i in range(num_p):
        p_input[i,0] =  f(x_input[i,0],y_input[i,0],z_input[i,0])
    pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(y_input).float(), requires_grad=True).to(device)
    pt_z_collocation = Variable(torch.from_numpy(z_input).float(), requires_grad=True).to(device)
    pt_p_collocation = Variable(torch.from_numpy(p_input).float(), requires_grad=False).to(device)
elif(resample_mode==5):
    from scipy.stats import qmc

    halton_sampler = qmc.Halton(d=3, scramble=True,seed=seed)
    samples = halton_sampler.random(n=num_p)
    x_input = samples[:, 0].reshape(-1, 1) * a
    y_input = samples[:, 1].reshape(-1, 1) * b
    z_input = samples[:, 2].reshape(-1, 1) * c
    print(x_input)
    p_input = np.zeros((num_p,1))
    for i in range(num_p):
        p_input[i,0] =  f(x_input[i,0],y_input[i,0],z_input[i,0])
    pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(y_input).float(), requires_grad=True).to(device)
    pt_z_collocation = Variable(torch.from_numpy(z_input).float(), requires_grad=True).to(device)
    pt_p_collocation = Variable(torch.from_numpy(p_input).float(), requires_grad=False).to(device)
elif(resample_mode==6):
    from pyDOE import lhs

    samples = lhs(n=3, samples=num_p)
    x_input = samples[:, 0].reshape(-1, 1) * a
    y_input = samples[:, 1].reshape(-1, 1) * b
    z_input = samples[:, 2].reshape(-1, 1) * c
    p_input = np.zeros((num_p,1))
    for i in range(num_p):
        p_input[i,0] =  f(x_input[i,0],y_input[i,0],z_input[i,0])
    pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(y_input).float(), requires_grad=True).to(device)
    pt_z_collocation = Variable(torch.from_numpy(z_input).float(), requires_grad=True).to(device)
    pt_p_collocation = Variable(torch.from_numpy(p_input).float(), requires_grad=False).to(device)
elif(resample_mode == 7 or resample_mode==2 or resample_mode==3):
    from scipy.stats import qmc
    sobel_sampler = qmc.Sobol(d=3, scramble=True,seed=seed)
    samples = sobel_sampler.random(n=num_p)
    x_input = samples[:, 0].reshape(-1, 1) * a
    y_input = samples[:, 1].reshape(-1, 1) * b
    z_input = samples[:, 2].reshape(-1, 1) * c
    p_input = np.zeros((num_p,1))
    for i in range(num_p):
        p_input[i,0] =  f(x_input[i,0],y_input[i,0],z_input[i,0])
    pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(y_input).float(), requires_grad=True).to(device)
    pt_z_collocation = Variable(torch.from_numpy(z_input).float(), requires_grad=True).to(device)
    pt_p_collocation = Variable(torch.from_numpy(p_input).float(), requires_grad=False).to(device)
elif(resample_mode==8):
    x_grid = np.linspace(0, a, 42)
    y_grid = np.linspace(0, b, 42)
    z_grid = np.linspace(0, c, 22)

    x_input, y_input, z_input = np.meshgrid(x_grid[:-1], y_grid[:-1], z_grid[:-1], indexing='ij')

    x_input = (x_input[:-1, :-1, :-1] + x_input[1:, 1:, 1:]) / 2
    y_input = (y_input[:-1, :-1, :-1] + y_input[1:, 1:, 1:]) / 2
    z_input = (z_input[:-1, :-1, :-1] + z_input[1:, 1:, 1:]) / 2

    x_input = x_input.reshape(-1, 1)
    y_input = y_input.reshape(-1, 1)
    z_input = z_input.reshape(-1, 1)
    print(x_input.shape[0])
    p_input = np.zeros((num_p,1))
    for i in range(num_p):
        p_input[i,0] =  f(x_input[i,0],y_input[i,0],z_input[i,0])
    pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(y_input).float(), requires_grad=True).to(device)
    pt_z_collocation = Variable(torch.from_numpy(z_input).float(), requires_grad=True).to(device)
    pt_p_collocation = Variable(torch.from_numpy(p_input).float(), requires_grad=False).to(device)

loss_history = []  # 用于存储损失值的列表




start = time.time()
for epoch in range(iterations):
    
    optimizer.zero_grad() # to make the gradients zero
    mse_f = 0
   
    #1:ASRR algorithm
    if resample_mode==1:
        if(epoch > adap_begin):
            max_indice = torch.argmax(h1*matrix_W)
            matrix_W[max_indice,0] /= (discrete_times**3)
            W_val = matrix_W[max_indice,0]
            W_add = W_val * torch.ones((discrete_times**3-1,1)).to(device)
            matrix_W = torch.cat((matrix_W,W_add), dim=0)

            max_x = x_add[max_indice,0]
            max_y = y_add[max_indice,0]
            max_z = z_add[max_indice,0]

            matrix_Size[max_indice,0] /= discrete_times
            Size_val = matrix_Size[max_indice,0]
            Size_add = Size_val * np.ones((discrete_times**3-1,1))
            matrix_Size = np.concatenate((matrix_Size, Size_add), axis = 0)

            x_new = np.arange(max_x, max_x + discrete_times*matrix_Size[max_indice,0]*a/x_grid - matrix_Size[max_indice,0]*a/x_grid/2, matrix_Size[max_indice,0]*a/x_grid)
                        
            y_new = np.arange(max_y, max_y + discrete_times*matrix_Size[max_indice,0]*b/y_grid - matrix_Size[max_indice,0]*b/y_grid/2, matrix_Size[max_indice,0]*b/y_grid)
            z_new = np.arange(max_z, max_z + discrete_times*matrix_Size[max_indice,0]*c/z_grid - matrix_Size[max_indice,0]*c/z_grid/2, matrix_Size[max_indice,0]*c/z_grid)
            ms_x_new, ms_y_new, ms_z_new  = np.meshgrid(x_new, y_new, z_new)
            ## Just because meshgrid is used, we need to do the following adjustment
            x_new = np.ravel(ms_x_new).reshape(-1,1)
            y_new = np.ravel(ms_y_new).reshape(-1,1)
            z_new = np.ravel(ms_z_new).reshape(-1,1)
                        
            z_add = np.concatenate((z_add,z_new[1:,:]), axis = 0)
            x_add = np.concatenate((x_add,x_new[1:,:]), axis = 0)
            y_add = np.concatenate((y_add,y_new[1:,:]), axis = 0)
                            
            num_p += (discrete_times**3-1)
        x_input = np.random.uniform(low=0, high=a/x_grid, size=(num_p,1)) * matrix_Size + x_add
        y_input = np.random.uniform(low=0, high=b/y_grid, size=(num_p,1)) * matrix_Size + y_add
        z_input = np.random.uniform(low=0, high=c/z_grid, size=(num_p,1)) * matrix_Size + z_add

        #draw the distribution of sampling points
        #if(epoch==iterations-1):
            #visualize_3d_points(x_input, y_input, z_input)        
        
        p_input = np.zeros((num_p,1))
        for i in range(num_p):
            p_input[i,0] =  f(x_input[i,0],y_input[i,0],z_input[i,0])         
        pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(y_input).float(), requires_grad=True).to(device)
        pt_z_collocation = Variable(torch.from_numpy(z_input).float(), requires_grad=True).to(device)
        pt_p_collocation = Variable(torch.from_numpy(p_input).float(), requires_grad=False).to(device)
        f_out, h1 = pde1(pt_x_collocation, pt_y_collocation, pt_z_collocation, pt_p_collocation, net1,0.853425,130/130,epoch)
                        
        loss = torch.sum(f_out * matrix_W) / (x_grid*y_grid*z_grid)
        print("epoch",epoch," : ", loss, "nump: ", num_p)
        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step()
        loss_history.append(loss.item()) 
                    
        with torch.autograd.no_grad():
            writer.add_scalar(tag="LOSS",
                            scalar_value=loss.item(),
                            global_step=epoch
                            )
        if (epoch + 1) % decay_interval == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate
        
        if (epoch == iterations-1):            
            for epoch_else in range(adap_end):
                optimizer.zero_grad() # to make the gradients zero
                f_out = pde(pt_x_collocation, pt_y_collocation, pt_z_collocation, pt_p_collocation, net1,0.853425,130/130)          
                loss = torch.sum(f_out * matrix_W) / (x_grid*y_grid*z_grid)
                loss.backward() # This is for computing gradients using backward propagation
                optimizer.step()
                loss_history.append(loss.item())
                            
                with torch.autograd.no_grad():
                    writer.add_scalar(tag="LOSS",
                                    scalar_value=loss.item(),
                                    global_step=epoch_else+iterations
                                    )
                if (epoch_else + 1) % 20 == 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= decay_rate
            

    #2:RAR from DeepONet
    elif resample_mode==2:
        train_first = 2025
        num_all = 10000
        num_choose = 999
        num_train = 1500
        if(epoch < train_first):
            
            f_out = pde(pt_x_collocation, pt_y_collocation, pt_z_collocation, pt_p_collocation, net1,0.853425,130/130)
            
            mse_f = (torch.sum(f_out)) / num_p
            #print("epoch",epoch," : ", mse_f)
        
            loss = mse_f
            loss.backward() # This is for computing gradients using backward propagation

            optimizer.step()
            with torch.autograd.no_grad():
                writer.add_scalar(tag="LOSS",
                                scalar_value=loss.item(),
                                global_step=epoch  
                                )

            if (epoch + 1) % decay_interval == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= decay_rate
        else:
            pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True).to(device)
            pt_y_collocation = Variable(torch.from_numpy(y_input).float(), requires_grad=True).to(device)
            pt_z_collocation = Variable(torch.from_numpy(z_input).float(), requires_grad=True).to(device)
            pt_p_collocation = Variable(torch.from_numpy(p_input).float(), requires_grad=False).to(device)
            f_out = pde(pt_x_collocation, pt_y_collocation, pt_z_collocation, pt_p_collocation, net1,0.853425,130/130)
                

            mse_f = (torch.sum(f_out)) / (x_input.shape[0])
            #print("epoch",epoch," : ", mse_f)
            
            loss = mse_f
            loss.backward() # This is for computing gradients using backward propagation

            optimizer.step()
            with torch.autograd.no_grad():
                writer.add_scalar(tag="LOSS",
                                scalar_value=loss.item(),
                                global_step=epoch  
                                )
            
            if (epoch + 1) % decay_interval == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= decay_rate
            
            if(epoch%num_train==0):
                x_input1 = np.random.uniform(low=0, high=a, size=(num_all,1))
                y_input1 = np.random.uniform(low=0, high=b, size=(num_all,1))
                z_input1 = np.random.uniform(low=0, high=c, size=(num_all,1))
                p_input1 = np.zeros((num_all,1))
                for i in range(num_all):
                    p_input1[i,0] =  f(x_input1[i,0],y_input1[i,0],z_input1[i,0])
                pt_x_collocation1 = Variable(torch.from_numpy(x_input1).float(), requires_grad=True).to(device)
                pt_y_collocation1 = Variable(torch.from_numpy(y_input1).float(), requires_grad=True).to(device)
                pt_z_collocation1 = Variable(torch.from_numpy(z_input1).float(), requires_grad=True).to(device)
                pt_p_collocation1 = Variable(torch.from_numpy(p_input1).float(), requires_grad=False).to(device)
                f_out = pde(pt_x_collocation1, pt_y_collocation1, pt_z_collocation1, pt_p_collocation1, net1,0.853425,130/130)
                point_key, point_idx = torch.topk(f_out,num_choose,dim=0)
                point_idx = point_idx.cpu().reshape(num_choose)
                x_input = np.concatenate((x_input,x_input1[point_idx]),axis=0)
                y_input = np.concatenate((y_input,y_input1[point_idx]),axis=0)
                z_input = np.concatenate((z_input,z_input1[point_idx]),axis=0)
                p_input = np.concatenate((p_input,p_input1[point_idx]),axis=0)
                
                if(x_input.shape[0] >= 32050):
                        break
        print("epoch",epoch," : ", loss, "nump: ", num_p)
    
    #3:RAD
    elif resample_mode==7:
        train_first = 2025
        num_all = 10000
        num_choose = 999
        num_train = 1500
        if(epoch < train_first):
                
            f_out = pde(pt_x_collocation, pt_y_collocation, pt_z_collocation, pt_p_collocation, net1,0.853425,130/130)
                
            mse_f = (torch.sum(f_out)) / num_p
            
            loss = mse_f
            loss.backward() # This is for computing gradients using backward propagation

            optimizer.step()
            with torch.autograd.no_grad():
                writer.add_scalar(tag="LOSS",
                                scalar_value=loss.item(),
                                global_step=epoch  
                                )

            if (epoch + 1) % decay_interval == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= decay_rate
        else:
                pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True).to(device)
                pt_y_collocation = Variable(torch.from_numpy(y_input).float(), requires_grad=True).to(device)
                pt_z_collocation = Variable(torch.from_numpy(z_input).float(), requires_grad=True).to(device)
                pt_p_collocation = Variable(torch.from_numpy(p_input).float(), requires_grad=False).to(device)
                f_out = pde(pt_x_collocation, pt_y_collocation, pt_z_collocation, pt_p_collocation, net1,0.853425,130/130)
                    

                mse_f = (torch.sum(f_out)) / (x_input.shape[0])
                
                loss = mse_f
                loss.backward() # This is for computing gradients using backward propagation

                optimizer.step()
                with torch.autograd.no_grad():
                    writer.add_scalar(tag="LOSS",
                                    scalar_value=loss.item(),
                                    global_step=epoch  
                                    )

                if (epoch + 1) % decay_interval == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= decay_rate
                
                if(epoch%num_train==0):
                    x_input1 = np.random.uniform(low=0, high=a, size=(num_all,1))
                    y_input1 = np.random.uniform(low=0, high=b, size=(num_all,1))
                    z_input1 = np.random.uniform(low=0, high=c, size=(num_all,1))
                    p_input1 = np.zeros((num_all,1))
                    for i in range(num_all):
                        p_input1[i,0] =  f(x_input1[i,0],y_input1[i,0],z_input1[i,0])
                    #p_input1 = calculate_power_density(XY, G, x_input1, y_input1,z_input1)
                    pt_x_collocation1 = Variable(torch.from_numpy(x_input1).float(), requires_grad=True).to(device)
                    pt_y_collocation1 = Variable(torch.from_numpy(y_input1).float(), requires_grad=True).to(device)
                    pt_z_collocation1 = Variable(torch.from_numpy(z_input1).float(), requires_grad=True).to(device)
                    pt_p_collocation1 = Variable(torch.from_numpy(p_input1).float(), requires_grad=False).to(device)
                    f_out = pde(pt_x_collocation1, pt_y_collocation1, pt_z_collocation1, pt_p_collocation1, net1,0.853425,130/130)
                    f_out = f_out / (torch.mean(f_out)) + 2
                    f_out = f_out / f_out.sum()
                    point_idx = torch.multinomial(f_out.squeeze(), num_choose, replacement=False).cpu()
                    
                    x_input = np.concatenate((x_input,x_input1[point_idx]),axis=0)
                    y_input = np.concatenate((y_input,y_input1[point_idx]),axis=0)
                    z_input = np.concatenate((z_input,z_input1[point_idx]),axis=0)
                    p_input = np.concatenate((p_input,p_input1[point_idx]),axis=0)
                    
                    if(x_input.shape[0] > 32050):
                        break
        print("epoch",epoch," : ", loss, "nump: ", num_p)

    #4/5/6/7/8:other non-adaptive sampling methods
    else:
        f_out = pde(pt_x_collocation, pt_y_collocation, pt_z_collocation, pt_p_collocation, net1,0.853425,130/130)
        
        mse_f = (torch.sum(f_out)) / num_p
        #print("epoch",epoch," : ", mse_f)
    
        loss = mse_f
        loss.backward() # This is for computing gradients using backward propagation
        print("epoch",epoch," : ", loss, "nump: ", num_p)
        optimizer.step()
        with torch.autograd.no_grad():
            writer.add_scalar(tag="LOSS",
                            scalar_value=loss.item(),
                            global_step=epoch  
                            )

        if (epoch + 1) % decay_interval == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate


end = time.time()
print("Solution time: {:.2f}s".format(end-start))


# Result plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


from matplotlib.ticker import FuncFormatter
def scale_formatter(x, pos):
    """Formatter to scale axis ticks"""
    return f'{x * 1000:.0f}'
def scale_formatter1(x, pos):
    """Formatter to scale axis ticks"""
    return f'{x * 1000:.1f}'
def visualize_3d_points(x, y, z):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    color = 'k'
    ax.scatter(x, y, z, c=color, s=2, alpha=0.3)
    vertices = np.array([
        [0, 0, 0],
        [0.015, 0, 0],
        [0.015, 0.015, 0],
        [0, 0.015, 0],
        [0, 0, 0.001],
        [0.015, 0, 0.001],
        [0.015, 0.015, 0.001],
        [0, 0.015, 0.001]
    ])
    faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]]
            ]
    

    cube = Poly3DCollection(faces, facecolor=(1, 1, 1, 0.1), edgecolor=(0,0,0,0.5), linewidths=2)
    ax.add_collection3d(cube)

    semi_transparent_edges = [
        [vertices[0], vertices[1]],  
        [vertices[0], vertices[3]], 
        [vertices[0], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[3], vertices[7]],
        [vertices[4], vertices[7]],
        [vertices[4], vertices[5]],
        [vertices[6], vertices[5]],
        [vertices[6], vertices[7]]
    ]

    trans_lines = Line3DCollection(semi_transparent_edges, colors=[(0, 0, 0, 1)], linewidths=2)
    ax.add_collection(trans_lines)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.grid(False)
    ax.set_xlabel('x (mm)', fontsize = 28, labelpad=15)
    ax.set_ylabel('y (mm)', fontsize = 28, labelpad=15)
    ax.set_zlabel('z (mm)', fontsize = 28, labelpad=20)
    ax.tick_params(axis='x', labelsize=28) 
    ax.tick_params(axis='y', labelsize=28)  
    ax.tick_params(axis='z', labelsize=28) 
    ax.zaxis.set_tick_params(pad=12)
    ax.set_xticks([0, 0.005,0.010, 0.015])
    ax.set_yticks([0, 0.005,0.010, 0.015])
    ax.set_zticks([0,  0.0005, 0.0010])
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(scale_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(scale_formatter))
    ax.zaxis.set_major_formatter(FuncFormatter(scale_formatter1))
    ax.set_xlim(0, 0.015)
    ax.set_ylim(0, 0.015)
    ax.set_zlim(0, 0.001)
    ax.view_init(elev=23, azim=-144)
    plt.savefig('pointshow1.png', dpi=600)
    plt.show()


fig = plt.figure(1)
ax = fig.add_axes(Axes3D(fig))

x=np.arange(0,a+5e-4,5e-4)
y=np.arange(0,b+5e-4,5e-4)
z = np.arange(0, c+1e-4, 1e-4)
ms_x, ms_y, ms_z = np.meshgrid(x, y, z)
## Just because meshgrid is used, we need to do the following adjustment
x = np.ravel(ms_x).reshape(-1,1)
y = np.ravel(ms_y).reshape(-1,1)
z = np.ravel(ms_z).reshape(-1,1)

p = np.zeros((10571,1))
for i in range(10571):
    p[i,0] =  f(x[i,0],y[i,0],z[i,0])
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)
pt_z = Variable(torch.from_numpy(z).float(), requires_grad=True).to(device)
pt_p = Variable(torch.from_numpy(p).float(), requires_grad=False).to(device)
    
f_out = pde(pt_x, pt_y, pt_z, pt_p, net1,0.853425,130/130)
f_out = f_out.data.cpu()
ms_f = f_out.reshape(ms_x.shape)
test_f_1=ms_f[:,:,2].numpy()
test_f_2=ms_f[:,:,5].numpy()
test_f_3=ms_f[:,:,8].numpy()


pt_u = net1(pt_x,pt_y,pt_z,0.853425,130/130)
u=pt_u.data.cpu()*T_std
ms_u = u.reshape(ms_x.shape)
test_u_1=ms_u[:,:,2].numpy()
test_u_2=ms_u[:,:,5].numpy()
test_u_3=ms_u[:,:,8].numpy()
x_draw=np.arange(0,a+5e-4,5e-4)
y_draw=np.arange(0,b+5e-4,5e-4)

x_draw, y_draw = np.meshgrid(x_draw, y_draw)
## Just because meshgrid is used, we need to do the following adjustment
surf = ax.plot_surface(x_draw,y_draw,test_u_1, cmap=cm.coolwarm)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)

fig2 = plt.figure(2,figsize=(10,10))
ax2 = fig2.add_axes(Axes3D(fig2))
surf2 = ax2.plot_surface(x_draw*1000,y_draw*1000,test_u_2, cmap=cm.coolwarm)
ax2.set_xlabel('x (mm)', fontsize=26, labelpad=15)
ax2.set_ylabel('y (mm)', fontsize=26, labelpad=15)
ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.set_zlabel('Temperature (K)', rotation=90, horizontalalignment='right', fontsize=26, labelpad=40)
ax2.set_xticks([0, 5, 10, 15])
ax2.set_yticks([0, 5, 10, 15])
ax2.tick_params(axis='x', labelsize=26)
ax2.tick_params(axis='y', labelsize=26)
ax2.tick_params(axis='z', labelsize=26)  
ax2.zaxis.set_tick_params(pad=20)
ax2.zaxis.set_major_locator(LinearLocator(5))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
cbar = fig2.colorbar(surf2, shrink=0.6, aspect=10, pad=0.14)
cbar.ax.tick_params(labelsize=26) 
plt.savefig('case4_1.png',dpi=600)

fig3 = plt.figure(3)
ax3 = fig3.add_axes(Axes3D(fig3))
surf3 = ax3.plot_surface(x_draw,y_draw,test_u_3, cmap=cm.coolwarm)
ax3.zaxis.set_major_locator(LinearLocator(10))
ax3.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig3.colorbar(surf3, shrink=0.5, aspect=5)

fig4 = plt.figure(4)

comsol_T_all = np.loadtxt('com3d_realchip_all_nl_0.0005.txt', usecols = 3)
comsol_T_all = comsol_T_all.reshape(11,31,31)
comsol_T_all = torch.from_numpy(comsol_T_all)
MAX_AE = 0
MEAN_AE = 0
for i in range(11):
    delta_T_all = torch.abs(ms_u[:,:,i] - comsol_T_all[i,:,:])
    MAX_AE = max(MAX_AE, torch.max(delta_T_all))
    MEAN_AE += torch.mean(delta_T_all)
MEAN_AE = MEAN_AE / 11    
RMAE = 100 * MEAN_AE / torch.max(comsol_T_all - T0)

comsol_T_0 = comsol_T_all[2,:,:].reshape(31,31)

delta_T_0 = comsol_T_0 - test_u_1
ax4 = fig4.add_axes(Axes3D(fig4))
surf4 = ax4.plot_surface(x_draw,y_draw,delta_T_0, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax4.zaxis.set_major_locator(LinearLocator(10))
ax4.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) 
plt.xlabel('x')
plt.ylabel('y')
fig4.colorbar(surf4, shrink=0.5, aspect=5)

comsol_T_1 = comsol_T_all[5,:,:].reshape(31,31)
delta_T_1 = comsol_T_1 - test_u_2
fig5 = plt.figure(5,figsize=(10,10))
ax5 = fig5.add_axes(Axes3D(fig5))
surf5 = ax5.plot_surface(x_draw*1000,y_draw*1000,delta_T_1, cmap=cm.coolwarm)
ax5.set_xlabel('x (mm)', fontsize=26, labelpad=15)
ax5.set_ylabel('y (mm)', fontsize=26, labelpad=15)
ax5.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  
ax5.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax5.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax5.set_zlabel('Temperature (K)', rotation=90, horizontalalignment='right', fontsize=26, labelpad=30)
ax5.set_xticks([0, 5, 10, 15])
ax5.set_yticks([0, 5, 10, 15])
ax5.tick_params(axis='x', labelsize=26) 
ax5.tick_params(axis='y', labelsize=26)  
ax5.tick_params(axis='z', labelsize=26) 
ax5.zaxis.set_tick_params(pad=15)
ax5.zaxis.set_major_locator(LinearLocator(5))
ax5.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
cbar = fig5.colorbar(surf5, shrink=0.6, aspect=10, pad=0.12)

cbar.ax.tick_params(labelsize=26) 
plt.savefig('case4_2.png',dpi=600)

fig6 = plt.figure(6)
   
comsol_T_2 = comsol_T_all[8,:,:].reshape(31,31)
delta_T_2 = comsol_T_2 - test_u_3
ax6 = fig6.add_axes(Axes3D(fig6))
surf6 = ax6.plot_surface(x_draw,y_draw,delta_T_2, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax6.zaxis.set_major_locator(LinearLocator(10))
ax6.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) 
plt.xlabel('x')
plt.ylabel('y')
fig6.colorbar(surf6, shrink=0.5, aspect=5)

fig7 = plt.figure(7)
az1 = fig7.add_axes(Axes3D(fig7))
surf7 = az1.plot_surface(x_draw,y_draw,comsol_T_0, cmap=cm.coolwarm,linewidth=0, antialiased=False)
az1.zaxis.set_major_locator(LinearLocator(10))
az1.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) 
         
plt.xlabel('x')
plt.ylabel('y') 
fig7.colorbar(surf7, shrink=0.5, aspect=5)

fig8 = plt.figure(8,figsize=(10,10))
az2 = fig8.add_axes(Axes3D(fig8))
surf8 = az2.plot_surface(x_draw*1000,y_draw*1000,comsol_T_1, cmap=cm.coolwarm)
az2.set_xlabel('x (mm)', fontsize=26, labelpad=15)
az2.set_ylabel('y (mm)', fontsize=26, labelpad=15)
az2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  
az2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
az2.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
az2.set_zlabel('Temperature (K)', rotation=90, horizontalalignment='right', fontsize=26, labelpad=40)
az2.set_xticks([0, 5, 10, 15])
az2.set_yticks([0, 5, 10, 15])
az2.tick_params(axis='x', labelsize=26)
az2.tick_params(axis='y', labelsize=26)
az2.tick_params(axis='z', labelsize=26)
az2.zaxis.set_tick_params(pad=20)
az2.zaxis.set_major_locator(LinearLocator(5))
az2.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
cbar = fig8.colorbar(surf8, shrink=0.6, aspect=10, pad=0.14)
cbar.ax.tick_params(labelsize=26)
plt.savefig('case4_3.png',dpi=600)

fig9 = plt.figure(9)
az3 = fig9.add_axes(Axes3D(fig9))
surf9 = az3.plot_surface(x_draw,y_draw,comsol_T_2, cmap=cm.coolwarm,linewidth=0, antialiased=False)
az3.zaxis.set_major_locator(LinearLocator(10))
az3.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) 
         
plt.xlabel('x')
plt.ylabel('y') 
fig9.colorbar(surf9, shrink=0.5, aspect=5)

figa = plt.figure(10)
axa = figa.add_axes(Axes3D(figa))
surfa = axa.plot_surface(x_draw,y_draw,test_f_1, cmap=cm.coolwarm)
axa.zaxis.set_major_locator(LinearLocator(10))
axa.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
figa.colorbar(surfa, shrink=0.5, aspect=5)

figb = plt.figure(11)
axb = figb.add_axes(Axes3D(figb))
surfb = axb.plot_surface(x_draw,y_draw,test_f_2, cmap=cm.coolwarm)
axb.zaxis.set_major_locator(LinearLocator(10))
axb.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
figb.colorbar(surfb, shrink=0.5, aspect=5)

figc = plt.figure(12)
axc = figc.add_axes(Axes3D(figc))
surfc = axc.plot_surface(x_draw,y_draw,test_f_3, cmap=cm.coolwarm)
axc.zaxis.set_major_locator(LinearLocator(10))
axc.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
figc.colorbar(surfc, shrink=0.5, aspect=5)

print("MAX AE: ", MAX_AE.item())
print("MEAN AE: ", MEAN_AE.item())
print("MEAN RELATIVE ERROR: ", RMAE.item(),"%")

plt.show() 