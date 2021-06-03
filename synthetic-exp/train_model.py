import torch
import numpy as np
from models import NN, NNPretrained

# train a one-hidden-layer neural network on given dataset

def MSE(Y, Y_pred):
    N = Y.shape[1]  # num of data
    return ((Y - Y_pred) ** 2).sum() / N

def train(X: torch.Tensor, Y: torch.Tensor, num_epoch, model: NN, early_stop_loss=0.5):
    for epoch in range(num_epoch):
        Y_pred = model.forward(X)
        loss = MSE(Y, Y_pred)
        if epoch % 50 == 0:
            print('loss at epoch {} is {}'.format(epoch, loss.item()))
        if loss < early_stop_loss:
            break
        loss.backward()
        model.optimize(momentum=0.9)

    Y_pred = model.forward(X)
    loss = MSE(Y, Y_pred)
    print('final loss at epoch {} is {}'.format(epoch, loss.item()))

def train_by_solving(X: torch.Tensor, Y: torch.Tensor, model: NNPretrained):
    # directly solve the convex program, instead of doing gradient descent
    with torch.no_grad():
        source_feature = model.sigmoid(model.W1.mm(X) + model.b1)
        SS = 0.
        YS = 0.
        s = source_feature.mean(dim=1).reshape([-1, 1])
        y = Y.mean(dim=1).reshape([-1, 1])
        N = X.shape[1]
        for i in range(N):
            SS += source_feature[:, i].reshape([-1, 1]).mm(source_feature[:, i].reshape([1, -1]))
            YS += Y[:, i].reshape([-1, 1]).mm(source_feature[:, i].reshape([1, -1]))
        SS /= N
        YS /= N
        P1 = YS - y.reshape([-1, 1]).mm(s.reshape([1, -1]))
        P2 = SS - s.reshape([-1, 1]).mm(s.reshape([1, -1]))
        model.W2 = P1.mm(P2.pinverse())
        model.b2 = y - (model.W2.mm(s)).reshape([-1, 1])

        Y_pred = model.forward(X)
        loss = MSE(Y, Y_pred)
        print('final loss is {}'.format(loss.item()))


if __name__ == "__main__":
    data_file_handle = "rbf-n-50-d-10-m-100-N-5000-num_centers-10"
    data_file = 'datasets/' + data_file_handle + '.npz'
    data = np.load(data_file)
    X = torch.Tensor(data['X'])
    Y = torch.Tensor(data['Y'])

    n = X.shape[0]  # input dim
    d = Y.shape[0]  # output dim
    # m = 60  # hidden layer dim
    lr = 0.2  # step size
    num_epoch = 10000  # max number of epoch

    m_begin = 50
    m_step = 10
    num_steps = 0
    m_stop = m_begin + num_steps * m_step + 1
    for m in range(m_begin, m_stop, m_step):
        model = NN(n=n, d=d, m=m, lr=lr)
        train(X, Y, num_epoch, model, early_stop_loss=0.)
        model_file_handle = 'rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(X.shape[1])
        model.save_model(model_file_handle)
