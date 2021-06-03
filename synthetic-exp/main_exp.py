import torch
from models import NN, NNPretrained
from transferability import Transferability
import numpy as np
from train_model import train, MSE, train_by_solving
import csv


seed = 233
torch.manual_seed(seed)
np.random.seed(seed)

data_file_handle = "rbf-n-50-d-10-m-100-N-5000-num_centers-10"
data_file = 'datasets/' + data_file_handle + '.npz'
data = np.load(data_file)
X = torch.Tensor(data['X'])
Y = torch.Tensor(data['Y'])

n = X.shape[0]  # input dim
d = Y.shape[0]  # output dim
N = X.shape[1]
m = 100  # target hidden layer width
m_source = 50 # source model hidden layer width
lr = 0.01  # step size
num_epoch = 50
level = 0  # level of the attack, should be with in 0 and n-1

model_file_handle = 'rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(N)
source_model_file_handle = 'rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m_source) + '-N-' + str(N)

f_T = NN(n=n, d=d, m=m, lr=lr)
f_T.load_model(model_file_handle)
f_S_clean = NN(n=n, d=d, m=m_source, lr=lr)
f_S_clean.load_model(source_model_file_handle)

W1 = f_S_clean.W1.clone().detach()
b1 = f_S_clean.b1.clone().detach()
W2 = f_S_clean.W2.clone().detach()
b2 = f_S_clean.b2.clone().detach()

# biased random deviation
W1_deviate = (torch.rand(W1.shape) - 0.5)
b1_deviate = (torch.rand(b1.shape) - 0.5)
W2_deviate = (torch.rand(W2.shape) - 0.5)
b2_deviate = (torch.rand(b2.shape) - 0.5)

begin_scale = 0
scale_step = 0.1
num_steps = 10

scale_list = []
alpha_S_T_list = []
alpha_T_S_list = []
loss_f_S_list = []
loss_f_T_list = []
gamma_S_T_list = []
gamma_T_S_list = []
alpha_gamma_combined_S_T_list = []
alpha_gamma_combined_T_S_list = []
gradient_loss_list = []

for i in range(num_steps):
    # deviate W1 and b1 to form f_S
    scale = begin_scale + i * scale_step
    W1 += scale * W1_deviate
    b1 += scale * b1_deviate
    W2 += scale * W2_deviate
    b2 += scale * b2_deviate
    f_S = NNPretrained(n=n, d=d, m=m, lr=lr, W1=W1, W2=W2, b1=b1, b2=b2)

    trans = Transferability(f_T=f_T, f_S=f_S, X=X, Y=Y, level=level)
    trans.compute_all()

    scale_list.append(scale)
    alpha_S_T_list.append(trans.alpha_S_T_mean)
    alpha_T_S_list.append(trans.alpha_T_S_mean)
    gamma_S_T_list.append(trans.gamma_S_T)
    gamma_T_S_list.append(trans.gamma_T_S)
    alpha_gamma_combined_S_T_list.append(trans.alpha_gamma_combined_S_T)
    alpha_gamma_combined_T_S_list.append(trans.alpha_gamma_combined_T_S)
    gradient_loss_list.append(trans.gradient_loss)

    train_by_solving(X, Y, f_S)
    Y_pred = f_S.forward(X)
    loss_f_S = MSE(Y, Y_pred).item() ** 0.5
    Y_pred = f_T.forward(X)
    loss_f_T = MSE(Y, Y_pred).item() ** 0.5
    loss_f_S_list.append(loss_f_S)
    loss_f_T_list.append(loss_f_T)

if m_source != m:
    results_path = 'results/result_scale-crossmodel-' + 'source_m-'+ str(m_source) + '-' + model_file_handle \
                   + '-attack_level-' + str(level) + '.csv'
else:
    results_path = 'results/result_scale-' + model_file_handle + '-attack_level-' + str(level) + '.csv'

results_content = ['scale', 'alpha_S_T', 'alpha_T_S', 'gamma_S_T', 'gamma_T_S', 'alpha_gamma_combined_S_T',
                   'alpha_gamma_combined_T_S', 'loss_f_S', 'loss_f_T', 'gradient_loss']

try:
    with open(results_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results_content)
        writer.writeheader()
        for i in range(num_steps):
            result = {'scale': scale_list[i], 'alpha_S_T': alpha_S_T_list[i], 'alpha_T_S': alpha_T_S_list[i],
                      'gamma_S_T': gamma_S_T_list[i], 'gamma_T_S': gamma_T_S_list[i],
                      'alpha_gamma_combined_S_T': alpha_gamma_combined_S_T_list[i],
                      'alpha_gamma_combined_T_S': alpha_gamma_combined_T_S_list[i], 'loss_f_S': loss_f_S_list[i],
                      'loss_f_T': loss_f_T_list[i], 'gradient_loss': gradient_loss_list[i]}
            writer.writerow(result)
except IOError:
    print("I/O error")

