import torch


def sigmoid_derivative(X: torch.Tensor):
    E = torch.exp(-X)
    return E / (1 + E).pow(2)


class NN:
    def __init__(self, n, d, m, lr):
        # n: input dim
        # d: output dim
        # m: hidden layer dim
        self.n = n
        self.d = d
        self.m = m
        self.W1 = torch.rand([m, n]) - 0.5
        self.b1 = torch.rand([m, 1]) - 0.5
        self.W2 = torch.rand([d, m]) - 0.5
        self.b2 = torch.rand([d, 1]) - 0.5
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.b1.requires_grad = True
        self.b2.requires_grad = True
        self.sigmoid = torch.nn.Sigmoid()
        self.lr = lr
        self.g_W1 = 0.
        self.g_b1 = 0.
        self.g_W2 = 0.
        self.g_b2 = 0.

    def forward(self, X, requires_grad=False):
        # X should be (n, N), where N is num of samples
        X.requires_grad = requires_grad
        Y_pred = self.W2.mm(self.sigmoid(self.W1.mm(X) + self.b1)) + self.b2
        return Y_pred

    def optimize(self, momentum=0.9):
        with torch.no_grad():
            self.g_W1 = momentum * self.g_W1 + self.lr * self.W1.grad
            self.g_b1 = momentum * self.g_b1 + self.lr * self.b1.grad
            self.g_W2 = momentum * self.g_W2 + self.lr * self.W2.grad
            self.g_b2 = momentum * self.g_b2 + self.lr * self.b2.grad
            self.W1 -= self.g_W1
            self.W2 -= self.lr * self.g_W2
            self.b1 -= self.lr * self.g_b1
            self.b2 -= self.lr * self.g_b2
            self.W1.grad.zero_()
            self.W2.grad.zero_()
            self.b1.grad.zero_()
            self.b2.grad.zero_()

    def save_model(self, save_handle):
        model_weights = {"W1": self.W1.detach(), "W2": self.W2.detach(), "b1": self.b1.detach(), "b2": self.b2.detach()}
        torch.save(model_weights, "models/model_weights-target-" + save_handle)

    def load_model(self, save_handle):
        model_weights = torch.load("models/model_weights-target-" + save_handle)
        self.W1 = model_weights['W1']
        self.W2 = model_weights['W2']
        self.b1 = model_weights['b1']
        self.b2 = model_weights['b2']
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.b1.requires_grad = True
        self.b2.requires_grad = True


    def jacobian(self, x: torch.Tensor):
        with torch.no_grad():
            return self.W2.mm(diagonalize(sigmoid_derivative(self.W1.mm(x.reshape([-1, 1])) + self.b1)).mm(self.W1))


class NNPretrained(NN):
    def __init__(self, n, d, m, lr, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor):
        # n: input dim
        # d: output dim
        # m: hidden layer dim
        super(NNPretrained, self).__init__(n, d, m, lr)
        self.W1 = W1.clone().detach()
        self.b1 = b1.clone().detach()
        self.W1.requires_grad = False
        self.b1.requires_grad = False
        self.W2 = W2.clone().detach()
        self.b2 = b2.clone().detach()
        self.W2.requires_grad = True
        self.b2.requires_grad = True

    def prepare_train(self):
        self.W2.requires_grad = True
        self.b2.requires_grad = True

    def optimize(self):
        with torch.no_grad():
            self.W2 -= self.lr * self.W2.grad
            self.b2 -= self.lr * self.b2.grad
            self.W2.grad.zero_()
            self.b2.grad.zero_()

    # def jacobian(self, x: torch.Tensor):
    #     with torch.no_grad():
    #         return diagonalize(sigmoid_derivative(self.W1.mm(x.reshape([-1, 1])) + self.b1)).mm(self.W1)


def diagonalize(x):
    n = len(x)
    D = torch.zeros([n, n])
    for i in range(n):
        D[i, i] = x[i]
    return D
