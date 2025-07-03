# -*- coding: utf-8 -*-
"""PINN_Pyrrhotite_Final.ipynb


import torch
import torch.nn as nn
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import seaborn as sns
from scipy.stats import norm


#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

N_train = 2500  #5000 Here maximum population is x*t = 64*40=2560
data = scipy.io.loadmat('/content/drive/MyDrive/Colab Notebooks/Completed/Pyrrhotite/Sulfate.mat')

P_star = data['s_star2']  # N x T  #ISA data
t_star = data['t']  # T x 1
X_star1 = data['X_star']  # N x 2

Q_star1 = data['s_star'] #Displacement data

#atmospheric o2 at the boundary
o2_atm = 9.26 #mol/m3 (9.26 mol/m3,  1 mol/m3 = 1e-6 mol/cm3)
#initial pyrrhotite concentration
fe_0 = 375.0 #mol/m3 (375.0 mol/m3,  1 mol/m3 = 1e-6 mol/cm3)
#initial ferrus concentration
fe2_0 = 10.0 #mol/m3 (10.0 mol/m3,  1 mol/m3 = 1e-6 mol/cm3)
#initial aluminate compound concentration
ca_0 = 100.0 #mol/m3 (100.0 mol/m3,  1 mol/m3 = 1e-6 mol/cm3)

Q_star11 = Q_star1/1000 #Convert mm data to m

#Deformation
m_fe = 17.58e-6 #m^3/mol #Molar volume: (17.58 cm^3/mol,  1 cm^3/mol = 1e-6 m^3/mol)
m_fe3 = 26.99e-6 #m^3/mol #Molar volume: (26.99 cm^3/mol,  1 cm^3/mol = 1e-6 m^3/mol)
m_ca = 678.0e-6 #m^3/mol #Molar volume: (678.0 cm^3/mol,  1 cm^3/mol = 1e-6 m^3/mol)
wi = 8 #stoichiometric coefficient accompanying the gypsum ###Assumed wi=8
m_gypsum = 74.2e-6 #m^3/mol #Molar volume: (74.2 cm^3/mol,  1 cm^3/mol = 1e-6 m^3/mol)
m_ettringite = 725.1e-6 #m^3/mol #Molar volume: (725.1 cm^3/mol,  1 cm^3/mol = 1e-6 m^3/mol)
f = 0.38 #The capillary porosity fraction
phi = 0.1 #Porosity 10%

alpha = 10e-6 #1/K Coefficient of thermal expansion in concrete
delT = 283 #K temperature range experienced by concrete
L = 29 #28.9m Length of the concrete domain with size (7.5cm*7.5cm*28.5 cm)

#Normalize data:
# calculate the mean and standard deviation
# normalize the data by subtracting the mean and dividing by the standard deviation

X_star = (X_star1 - np.mean(X_star1,axis=0)) /np.std(X_star1,axis=0)
#When axis=0, np.mean and np.std functions calculate the mean and standard deviation of each column in the input array X_star1, which represents the features of the data.
#So, mean and std become arrays with shape (n_features,), where each element corresponds to the mean and standard deviation of a particular feature across all samples.
X_star[np.isnan(X_star)] = np.nanmean(X_star) # the transformed data has zero mean and unit variance


P_star = (P_star - np.min(P_star)) / (np.max(P_star) - np.min(P_star)) #Min-max scaler normalized
Q_star = (Q_star11 - np.min(Q_star11)) / (np.max(Q_star11) - np.min(Q_star11)) #Min-max scaler normalized

fe_0 = fe_0/375 #Normalize bcoz fe is compared to normalized ISA data for loss calculation

N = X_star.shape[0]
T = t_star.shape[0]

def calculate_D(D_0, theta, gamma):
    # Calculate the intermediate terms
    term1 = theta * torch.exp(gamma)
    term2 = theta * (torch.exp(gamma) - 1)
    denominator = 1 + term2

    # Calculate D
    D = D_0 * (1 - term1 / denominator)
    return D

# Construct the model with the desired t value:
for i, tt in enumerate(t_star):
  for j, x in enumerate(X_star[:,0]): 
    for k, y in enumerate(X_star[:,1]):

        class Pyrrhotite():

            def __init__(self, X, Y, T, ISA, disp, lammda, theta0):

                self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
                self.y = torch.tensor(Y, dtype=torch.float32, requires_grad=True)
                self.t = torch.tensor(T, dtype=torch.float32, requires_grad=True)

                self.ISA = torch.tensor(ISA, dtype=torch.float32)
                self.disp = torch.tensor(disp, dtype=torch.float32)
                self.lammda = torch.tensor(lammda, dtype=torch.float32)
                self.theta0 = torch.tensor(theta0, dtype=torch.float32)
           
                #null vector to test against f and g:
                self.null = torch.zeros((self.x.shape[0], 1))

                # initialize network:
                self.network()

                self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=0.003, max_iter=200000, max_eval=50000,
                                                  history_size=50, tolerance_grad=1e-05, tolerance_change=0.5 * np.finfo(float).eps,
                                                  line_search_fn="strong_wolfe")

                self.mse = nn.MSELoss()

                #loss
                self.ls = 0

                #iteration number
                self.iter = 0
                self.losses = []

            def network(self):
                n=64 #64 Normalize the data helps to reduce the complexity of network
                self.net = nn.Sequential(
                    nn.Linear(5, n), nn.Tanh(), #4 inputs (x,y,t,lammda, theta0,L)
                    #nn.Dropout(p=0.5), # add dropout layer
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, 8)) #8 outputs

               

            def function(self, x, y, t, lammda, theta0):

                res = self.net(torch.hstack((x, y, t, lammda, theta0)))               
                o2, fe, fe2, fe3, so4, s, ca, disp = res[:, 0:1], res[:, 1:2], res[:, 2:3], res[:, 3:4], res[:, 4:5], res[:, 5:6], res[:, 6:7], res[:, 7:8]

                if i == 0:
                  o2[:, :] = 0.0
                  fe[:, :] = fe_0
                  fe2[:, :] = fe2_0
                  fe3[:, :] = 0.0
                  so4[:, :] = 0.0
                  s[:, :] = 0.0
                  ca[:, :] = ca_0
                  disp[:, :] = 0.0

                # Add boundary conditions for all time steps except the first one (zero)
                if i > 0:
                  o2[:, :][x == x[0]]=o2_atm
                  o2[:, :][x == x[-1]]=o2_atm
                  o2[:, :][y == y[0]]=o2_atm
                  o2[:, :][y == y[-1]]=o2_atm

                  s[:, :][x == x[0]]=so4[-1,:]
                  s[:, :][x == x[-1]]=so4[-1,:]
                  s[:, :][y == y[0]]=so4[-1,:]
                  s[:, :][y == y[-1]]=so4[-1,:]
                  

                if i == 60:   #can use len(t) - 1 if t_traina nd t_test are same size

                  fe[:, :] = 0.0
                  fe2[:, :] = 0.0

                # store the boundary values separately
                o2_ic = (o2[-1, :]).detach().numpy()
                o2_left = (o2[:, :][x == x[0]]).detach().numpy()
                o2_right = (o2[:, :][x == x[-1]]).detach().numpy()
                o2_bottom = (o2[:, :][y == y[0]]).detach().numpy()
                o2_top = (o2[:, :][y == y[-1]]).detach().numpy()
                o2_BC=np.concatenate([o2_ic, o2_left, o2_right, o2_bottom, o2_top])
                o2_BC = torch.from_numpy(o2_BC)

                fe_BC = (fe[-1, :]).detach().numpy() #includes both IC and BC
                fe_BC=np.concatenate([fe_BC])
                fe_BC = torch.from_numpy(fe_BC)

                fe2_BC = (fe2[-1, :]).detach().numpy() #includes both IC and BC
                fe2_BC=np.concatenate([fe2_BC])
                fe2_BC = torch.from_numpy(fe2_BC)

                fe3_BC = (fe3[-1, :]).detach().numpy() #includes both IC and BC
                fe3_BC=np.concatenate([fe3_BC])
                fe3_BC = torch.from_numpy(fe3_BC)

                so4_BC = (so4[-1, :]).detach().numpy() #includes both IC and BC
                so4_BC=np.concatenate([so4_BC])
                so4_BC = torch.from_numpy(so4_BC)

                s_ic = (s[-1, :]).detach().numpy()
                s_left = (s[:, :][x == x[0]]).detach().numpy()
                s_right = (s[:, :][x == x[-1]]).detach().numpy()
                s_bottom = (s[:, :][y == y[0]]).detach().numpy()
                s_top = (s[:, :][y == y[-1]]).detach().numpy()
                s_BC=np.concatenate([s_ic, s_left, s_right, s_bottom, s_top])
                s_BC = torch.from_numpy(s_BC)

                ca_BC = (ca[-1, :]).detach().numpy() #includes both IC and BC
                ca_BC=np.concatenate([ca_BC])
                ca_BC = torch.from_numpy(ca_BC)

                #Derivatives
                o2_x = torch.autograd.grad(o2, x, grad_outputs=torch.ones_like(o2), create_graph=True)[0]
                o2_xx = torch.autograd.grad(o2_x, x, grad_outputs=torch.ones_like(o2_x), create_graph=True)[0]
                o2_y = torch.autograd.grad(o2, y, grad_outputs=torch.ones_like(o2), create_graph=True)[0]
                o2_yy = torch.autograd.grad(o2_y, y, grad_outputs=torch.ones_like(o2_y), create_graph=True)[0]
                o2_t = torch.autograd.grad(o2, t, grad_outputs=torch.ones_like(o2), create_graph=True)[0]

                fe_t = torch.autograd.grad(fe, t, grad_outputs=torch.ones_like(fe), create_graph=True)[0]

                fe2_t = torch.autograd.grad(fe2, t, grad_outputs=torch.ones_like(fe2), create_graph=True)[0]

                fe3_t = torch.autograd.grad(fe3, t, grad_outputs=torch.ones_like(fe3), create_graph=True)[0]

                so4_t = torch.autograd.grad(so4, t, grad_outputs=torch.ones_like(fe3), create_graph=True)[0]

                s_x = torch.autograd.grad(s, x, grad_outputs=torch.ones_like(s), create_graph=True)[0]
                s_xx = torch.autograd.grad(s_x, x, grad_outputs=torch.ones_like(s_x), create_graph=True)[0]
                s_y = torch.autograd.grad(s, y, grad_outputs=torch.ones_like(s), create_graph=True)[0]
                s_yy = torch.autograd.grad(s_y, y, grad_outputs=torch.ones_like(s_y), create_graph=True)[0]
                s_t = torch.autograd.grad(s, t, grad_outputs=torch.ones_like(s), create_graph=True)[0]

                ca_t = torch.autograd.grad(ca, t, grad_outputs=torch.ones_like(ca), create_graph=True)[0]

                gamma=0.3
                alpha=0.5
                alpha = torch.tensor(alpha)

                theta=theta0+(1-theta0)*(1-torch.exp(-alpha*lammda))
                D_0 = 15.77 #m2/year (5e-7 m²/s,   1m2/s=3.154e+7 m2/year)
                D_0 = torch.tensor(D_0)
                theta = torch.tensor(theta)
                gamma = torch.tensor(gamma)

                D_o2 = calculate_D(D_0, theta, gamma)
                D_s = 0.00009462 #m2/year (3.0e-12 m²/s,   1m2/s=3.154e+7 m2/year)
                g = 0.25
                k2 = 4.9833 # m3/(year.mol) (1.58e-7 m³/(s·mol),  1m3/s=3.154e+7 m3/year)
                k4 = 0.000899 # m3/(year.mol) (2.85e-11 m³/(s·mol),  1m3/s=3.154e+7 m3/year)
                k = 0.1095 # m3/(year.mol) (3e-14 m³/(mol.day),  1m3/d=365 m3/year) ###Should be 1.12×10-4 [m3/(mole day)]]
                x_=0.1
                delta = 9-3*x_
                rho = 8-2*x_
                q = (9-3*x_)/4

                f_o2 = o2_t - (o2_xx + o2_yy)*D_o2 +g*k2*o2*fe2  

                f_fe = fe_t +k4*fe*fe3 #fe
                f_fe2 = fe2_t - delta*k4*fe*fe3 + k2*o2*fe2   
                f_fe3 = fe3_t - k2*o2*fe2 + rho*k4*fe*fe3    
                f_so4 = so4_t - k4*fe*fe3     #so4
                f_s = s_t - (s_xx + s_yy)*D_s + k*ca*s  
                f_ca = ca_t + (k/q)*ca*s     


                delvv_fe3 = fe*((1-x_)*m_fe3 - m_fe) #volumetric deformation due to iron hydroxide (fe)
                delvv_ettr = (ca)*(m_ca + wi*m_gypsum - m_ettringite) #volumetric deformation for ettringite (sulfate attack)
                delvv_total = delvv_fe3 + delvv_ettr # Total volumetric deformation
                delvv_inf = abs(delvv_total-f*phi) #Final volumetric expansion or maximum expansion
                delvv_t = delvv_inf*((fe_0 - fe)/fe_0) # Volumetric expansion at time t

                return o2, fe, fe2, fe3, so4, s, ca, disp, delvv_t, f_o2, f_fe, f_fe2, f_fe3, f_so4, f_s, f_ca, o2_BC, fe_BC, fe2_BC, fe3_BC, so4_BC, s_BC, ca_BC


            def closure(self):
                # reset gradients to zero:
                self.optimizer.zero_grad()

                # predictions:
                o2_prediction, fe_prediction, fe2_prediction, fe3_prediction, so4_prediction, s_prediction, ca_prediction, disp_prediction, delvv_t,\
                f_o2_prediction, f_fe_prediction, f_fe2_prediction, f_fe3_prediction, f_so4_prediction, f_s_prediction, f_ca_prediction,\
                o2_BC_prediction, fe_BC_prediction, fe2_BC_prediction, fe3_BC_prediction, so4_BC_prediction, s_BC_prediction, ca_BC_prediction = self.function(self.x, self.y, self.t, self.lammda, self.theta0)

                # calculate losses
                #Data loss:
                disp_p = delvv_t*alpha*delT*L #numerical findings from prediction of delvv_t
                disp_loss = self.mse(disp_prediction + disp_p, self.disp) #Combining the NN prediction and numerical findings

                fe_target = (1-self.ISA)*fe_0
                fe_loss = self.mse(fe_prediction, fe_target)

                #PDE loss:
                f_o2_loss = self.mse(f_o2_prediction, self.null)
                f_fe_loss = self.mse(f_fe_prediction, self.null)
                f_fe2_loss = self.mse(f_fe2_prediction, self.null)
                f_fe3_loss = self.mse(f_fe3_prediction, self.null)
                f_so4_loss = self.mse(f_so4_prediction, self.null)
                f_s_loss = self.mse(f_s_prediction, self.null)
                f_ca_loss = self.mse(f_ca_prediction, self.null)
                #BC loss:
                o2_BC_loss = self.mse(o2_prediction[-1,:], o2_BC_prediction)
                fe_BC_loss = self.mse(fe_prediction[-1,:], fe_BC_prediction)
                fe2_BC_loss = self.mse(fe2_prediction[-1,:], fe2_BC_prediction)
                fe3_BC_loss = self.mse(fe3_prediction[-1,:], fe3_BC_prediction)
                so4_BC_loss = self.mse(so4_prediction[-1,:], so4_BC_prediction)
                s_BC_loss = self.mse(s_prediction[-1,:], s_BC_prediction)
                ca_BC_loss = self.mse(ca_prediction[-1,:], ca_BC_prediction)

                # L2 regularization
                l2_reg = torch.tensor(0.0)
                for param in self.net.parameters():
                  l2_reg += torch.norm(param, p=2)

                self.ls = (disp_loss+fe_loss) + (f_o2_loss + f_fe_loss + f_fe2_loss + f_fe3_loss + f_so4_loss + f_s_loss + f_ca_loss) + (o2_BC_loss + fe_BC_loss + fe2_BC_loss + fe3_BC_loss + so4_BC_loss + s_BC_loss + ca_BC_loss)
                #self.ls = (disp_loss+fe_loss) + (f_o2_loss + f_fe_loss + f_fe2_loss + f_fe3_loss + f_so4_loss + f_s_loss + f_ca_loss) + (o2_BC_loss + fe_BC_loss + fe2_BC_loss + fe3_BC_loss + so4_BC_loss + s_BC_loss + ca_BC_loss) + 0.01*l2_reg  # Add L2 regularization term
                # derivative with respect to net's weights:
                self.ls.backward()

                self.iter += 1
                if not self.iter % 1:
                    loss = self.ls.item() # save the current loss value
                    self.losses.append(loss) # add the loss value to the list
                    print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.ls))

                return self.ls

            def train(self):

                # training loop
                self.net.train()
                self.optimizer.step(self.closure)

                plt.plot(range(len(self.losses)), self.losses)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Loss')
                plt.show()

# Rearrange Data
XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T
lamda_star = np.full((64, 1), -0)
LL = np.tile(lamda_star, (1, N)).T  # N x T
theta0_star = np.full((64, 1), 0.85)
MM = np.tile(theta0_star, (1, N)).T  # N x T
L_star = np.full((64, 1), 25)
NN = np.tile(L_star, (1, N)).T  # N x T

PP = P_star  # N x
QQ = Q_star

x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1
lamda = LL.flatten()[:, None]  # NT x 1
theta0 = MM.flatten()[:, None]  # NT x 1
#L = NN.flatten()[:, None]  # NT x 1

p = PP.flatten()[:, None]  # NT x 1
q = QQ.flatten()[:, None]  # NT x 1

# Training Data
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
p_train = p[idx, :]
q_train = q[idx, :]
lamda_train = lamda[idx, :]
theta0_train = theta0[idx, :]
#L_train = L[idx, :]

#Plot collocation point
XX1 = np.tile(X_star1[:, 0:1], (1, T))  # N x T
YY1 = np.tile(X_star1[:, 1:2], (1, T))  # N x T
X_points, Y_points = np.meshgrid(XX1*100,YY1*100) #m into cm
X_points = X_points[idx, :]
Y_points = Y_points[idx, :]

fig_1 = plt.figure(1, figsize=(30, 5))
plt.plot(X_points, Y_points,'*', color = 'blue', markersize = 0.1, label = 'Boundary collocation points= 100')

plt.xlabel(r'$x (cm)$')
plt.ylabel(r'$y (cm)$')
plt.title('Collocation points')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis('scaled')
plt.show()

#fig_1.savefig('collocation_points_Helmholtz.png', dpi = 500)

#Run model
#'''
pinn = Pyrrhotite(x_train, y_train, t_train, p_train, q_train, lamda_train, theta0_train)
pinn.train()
torch.save(pinn.net.state_dict(), 'model.pt')
#'''
#load model
pinn = Pyrrhotite(x_train, y_train, t_train, p_train, q_train, lamda_train, theta0_train)
pinn.net.load_state_dict(torch.load('model.pt'))
pinn.net.eval()

# Test Data
# predict data to actual scale
X_star2 = (X_star * np.std(X_star1,axis=0)) + np.mean(X_star1,axis=0)
x_test = X_star2[:, 0:1]
y_test = X_star2[:, 1:2]
lamda_test1 = np.full((64,1), 0) #Pick a value other than lamda=0m, to test on [USe -sign behind (say -3) to show correct plot]
theta0_test1 = np.full((64,1), 0.85) #Pick a value other than theta0=0.85, to test on
L_test1 = np.full((64,1), 25) #Pick a value other than L=25m, to test on

q_test = Q_star1[:, -1].reshape(-1, 1) #last time step

# Time values
t_start = 0.0  # Start time
t_end = 60.0 #60 or 80  # End time
num_times = 60  # Number of time instances
t_test = np.zeros((x_test.shape[0], x_test.shape[1])) + np.linspace(t_start, t_end, num_times)  # Array of time values
t_test = t_test.transpose(-1,0)

x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True)
lamda_test = torch.tensor(lamda_test1, dtype=torch.float32, requires_grad=False)
theta0_test = torch.tensor(theta0_test1, dtype=torch.float32, requires_grad=False)
L_test = torch.tensor(L_test1, dtype=torch.float32, requires_grad=False)

fe_predict=[]
ISA_predict=[]
disp_predict=[]
for i in range(num_times):
    t_test_i = t_test[i].reshape(-1, 1)
    t_test_i = torch.tensor(t_test_i, dtype=torch.float32, requires_grad=True)
    o2_prediction, fe_prediction, fe2_prediction, fe3_prediction, so4_prediction, s_prediction, ca_prediction, disp_prediction, delvv_t, f_o2_prediction, f_fe_prediction, f_fe2_prediction, f_fe3_prediction, f_so4_prediction, f_s_prediction, f_ca_prediction, o2_BC_prediction, fe_BC_prediction, fe2_BC_prediction, fe3_BC_prediction, so4_BC_prediction, s_BC_prediction, ca_BC_prediction = pinn.function(x_test, y_test, t_test_i, lamda_test, theta0_test)
    ISA_prediction = (1-fe_prediction/fe_0)*100 #in %
    fe_prediction = fe_prediction*375
    disp_prediction = disp_prediction * (np.max(Q_star11) - np.min(Q_star11)) + np.min(Q_star11) #Normalized to original
    disp_prediction = disp_prediction*1000 #m to mm

    fe_predict.append(fe_prediction) #get time-dependent results for the variable
    ISA_predict.append(ISA_prediction) #get time-dependent results for the variable
    disp_predict.append(disp_prediction) #get time-dependent results for the

#Time-dependent plot:

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg

#calculate avg of all z axes values over the domain
#Pick the desired variable to see prediction values:
arr_1 = ISA_predict
zzz=[]
for i in range(num_times):
    p=cal_average(arr_1[i])
    zzz.append(p)
zzz=torch.tensor(zzz)
zzz = torch.clamp(zzz, max=100)

#calculate avg of all z axes values over the domain
#Pick the desired variable to see prediction values:
arr_2 = disp_predict
zzz2=[]
for i in range(num_times):
    p=cal_average(arr_2[i])
    zzz2.append(p)
zzz2=torch.tensor(zzz2)
zzz2 = torch.clamp(zzz2, min=0)

#Convergence plot
#Plot L2 norm error:
actual1 = (P_star[0,:]*100).reshape(-1, 1) #ISA
h1=[]
for i in range(len(t_test)):
    predict1=arr_1[i].detach().numpy()
    error1_l2 = np.linalg.norm(actual1[i]-predict1[i],2)/np.linalg.norm(actual1[i],2)
    h1.append(error1_l2)

actual2 = (Q_star1[0,:]).reshape(-1, 1) #disp
h2=[]
for i in range(len(t_test)):
    predict2=arr_2[i].detach().numpy()
    error2_l2 = np.linalg.norm(actual2[i]-predict2[i],2)/np.linalg.norm(actual2[i],2)
    h2.append(error2_l2)

plt.plot(t_test,h1,'c-', markersize = 4, label = 'ISA Progress') #ISA
plt.plot(t_test,h2, 'g-', markersize = 4, label = 'Displacement') #disp

plt.xlabel(r'Time (years)',fontsize=12)
plt.ylabel(r'$L_2$ error')
plt.title(r'$L_2$ Norm of the error', fontsize=12)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)

# Create the legend with colored labels on the top left
plt.legend(loc='upper right', labels=['ISA Progress', 'Displacement'])
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('cyan')
leg.legendHandles[1].set_color('green')
plt.grid()
plt.show()

#Validation

# Assuming Q_star1 is a NumPy array
observed_values1 = P_star[0, :]*100 #ISA

# Convert TensorFlow Tensor to NumPy array
predicted_values1 = zzz.numpy()

# Calculate residuals
residuals1 = observed_values1 - predicted_values1

# Create histogram
plt.figure(figsize=(10, 6))
sns.histplot(residuals1, kde=True, stat="density", color="skyblue", bins=20, label="Residuals Histogram")

# Plot the assumed distribution (normal distribution in this case)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, np.mean(residuals1), np.std(residuals1))
plt.plot(x, p, 'k', linewidth=2, label='Assumed Distribution (Normal)')

plt.xlabel('Residuals',fontsize=15)
plt.ylabel('Density',fontsize=15)
#plt.title('Histogram and Assumed Distribution of Residuals')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

# Assuming Q_star1 is a NumPy array
observed_values2 = Q_star1[0, :] #disp

# Convert TensorFlow Tensor to NumPy array
predicted_values2 = zzz2.numpy()

# Calculate residuals
residuals2 = observed_values2 - predicted_values2

# Create histogram
plt.figure(figsize=(10, 6))
sns.histplot(residuals2, kde=True, stat="density", color="skyblue", bins=20, label="Residuals Histogram")

# Plot the assumed distribution (normal distribution in this case)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, np.mean(residuals2), np.std(residuals2))
plt.plot(x, p, 'k', linewidth=2, label='Assumed Distribution (Normal)')

plt.xlabel('Residuals',fontsize=15)
plt.ylabel('Density',fontsize=15)
#plt.title('Histogram and Assumed Distribution of Residuals')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

#ISA predict:

#Calculate the error between oci_orihginal and predicted oci (here zzz1)
# Calculate R-squared (R2)
r2 = r2_score(P_star[0,:]*100, zzz)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(P_star[0,:]*100, zzz))

# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(actual, predicted):
    return torch.mean(torch.abs((actual - predicted) / actual)) * 100

mape = calculate_mape(torch.tensor(P_star[0,:]*100), zzz)

print("R-squared:", r2)
print("RMSE:", rmse)

# Plot ISA prediction vs observation:

plt.plot(P_star[0,:]*100, zzz, 'rs', markersize=4) #ISA
a=[0,100]
b=[0,100]
plt.plot(a, b, 'b-', markersize=4)

# Annotate the plot with R2, RMSE, and MAPE values
plt.text(68.8, 16, f'R² = {r2:.2f}', fontsize=10, color='black')
plt.text(68.8, 12, f'RMSE = {rmse:.2f}', fontsize=10, color='black')

# Set plot labels and title with increased fontsize
plt.xlabel('Observed ISA progress (%)', fontsize=15)
plt.ylabel('Predicted ISA progress (%)', fontsize=15)
plt.ylim(bottom=0)
plt.ylim(-5, 105)
plt.xlim(-5, 105)
# Increase the size of axis values
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()

#Disp predict:

# Calculate R-squared (R2)
r2 = r2_score(Q_star1[0,:], zzz2)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(Q_star1[0,:], zzz2))

print("R-squared:", r2)
print("RMSE:", rmse)

# Plot disp prediction vs observation:

plt.plot(Q_star1[0,:], zzz2, 'rs', markersize=4) #disp.
a=[0,260]
b=[0,260]
plt.plot(a, b, 'b-', markersize=4)

# Annotate the plot with R2, RMSE, and MAPE values
plt.text(175.8, 43, f'R² = {r2:.2f}', fontsize=10, color='black')
plt.text(175.8, 29, f'RMSE = {rmse:.2f}', fontsize=10, color='black')

# Set plot labels and title with increased fontsize
plt.xlabel('Observed displacement (mm)', fontsize=15)
plt.ylabel('Predicted displacement (mm)', fontsize=15)
plt.ylim(bottom=0)
plt.ylim(-5, 265)
plt.xlim(-5, 265)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()