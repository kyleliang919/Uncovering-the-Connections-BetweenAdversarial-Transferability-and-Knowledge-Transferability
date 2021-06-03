import torch
from models import NN, NNPretrained


# refer to the paper for notations
class Transferability:
    def __init__(self, f_T: NN, f_S: NN, X: torch.Tensor, Y: torch.Tensor, level=0):
        self.f_T = f_T
        self.f_S = f_S
        self.X = X
        self.Y = Y
        self.N = X.shape[1]  # num of data
        self.n = f_T.n
        self.d = f_T.d
        #self.m = f_S.m
        self.m = f_S.d
        self.level = level # use level(th) attack; should be with in 0 and n-1
        self.jacob_T = None
        self.jacob_S = None
        self.jacob_T_2_norm = None
        self.jacob_S_2_norm = None
        self.jacob_T_D_2_norm = None
        self.jacob_S_D_2_norm = None
        self.delta_T = None
        self.delta_S = None
        #self.tau1 = None
        #self.tau1_mean = None
        #self.tau2 = None
        self.Delta_T_S = None
        self.Delta_S_S = None
        self.Delta_S_T = None
        self.Delta_T_T = None
        #self.lambda_S = None
        #self.lambda_T = None
        #self.lambda_S_max = None
        #self.lambda_T_max = None
        self.alpha_S_T = None
        self.alpha_S_T_mean = None
        self.alpha_T_S = None
        self.alpha_T_S_mean = None
        self.A = None
        #self.theorem_2_upper_bound = None
        #self.theorem_2_LHS = None
        self.beta = None
        self.gamma_S_T = None
        self.gamma_S_T_mean = None
        self.gamma_T_S = None
        self.gamma_T_S_mean = None
        self.alpha_gamma_combined_S_T = None
        self.alpha_gamma_combined_T_S = None
        self.gradient_loss = None
        #self.theorem_3_upper_bound = None

    def compute_all(self):
        print('computing Jacobian...')
        self.compute_jacobian()
        print('computing delta, Delta')
        self.compute_delta_Delta()
        print('computing alpha...')
        self.compute_alpha()
        print('computing gamma...')
        self.compute_gamma()
        print('computing gradient loss...')
        self.compute_gradient_loss()
        # print('compute tau1...')
        # self.compute_tau1()
        # print('compute tau2...')
        # self.compute_tau2()
        # print('compute theorem 2...')
        # self.compute_theorem_2()
        # print('compute beta...')
        # self.compute_beta()
        print('done!')

    def compute_jacobian(self):
        self.jacob_T = torch.zeros([self.d, self.n, self.N])
        self.jacob_S = torch.zeros([self.m, self.n, self.N])
        #self.jacob_S = torch.zeros([self.d, self.n, self.N])
        for n in range(self.N):
            x = self.X[:, n]  # n-th data point
            self.jacob_T[:, :, n] = self.f_T.jacobian(x)
            self.jacob_S[:, :, n] = self.f_S.jacobian(x)

    def compute_delta_Delta(self):
        # prerequisite: compute_jacobian
        self.delta_T = torch.zeros([self.n, self.N])
        self.delta_S = torch.zeros([self.n, self.N])
        self.Delta_T_S = torch.zeros([self.m, self.N])
        self.Delta_S_S = torch.zeros([self.m, self.N])
        self.Delta_S_T = torch.zeros([self.d, self.N])
        self.Delta_T_T = torch.zeros([self.d, self.N])
        #self.Delta_S_S = torch.zeros([self.d, self.N])
        self.lambda_S = torch.zeros(self.N)
        self.lambda_T = torch.zeros(self.N)
        self.jacob_T_2_norm = torch.zeros(self.N)
        self.jacob_S_2_norm = torch.zeros(self.N)
        deviation = 0
        for i in range(self.N):
            # compute for source model
            u, s, v = torch.svd(self.jacob_S[:, :, i])
            sorted_s, indices = s.abs().sort(descending=True)
            self.delta_S[:, i] = v[:, indices[self.level]]
            # self.lambda_S[i] = sorted_s[1] / sorted_s[0]
            self.jacob_S_2_norm[i] = sorted_s[0].item()
            self.Delta_S_S[:, i] = self.jacob_S[:, :, i].mm(self.delta_S[:, i].reshape([-1, 1])).squeeze()
            self.Delta_S_T[:, i] = self.jacob_T[:, :, i].mm(self.delta_S[:, i].reshape([-1, 1])).squeeze()

            # compute for target model
            u, s, v = torch.svd(self.jacob_T[:, :, i])
            sorted_s, indices = s.abs().sort(descending=True)
            self.delta_T[:, i] = v[:, indices[self.level]]
            self.lambda_T[i] = sorted_s[1] / sorted_s[0]
            self.jacob_T_2_norm[i] = sorted_s[0].item()
            #deviation += sorted_s[0].item() ** 2
            self.Delta_T_T[:, i] = self.jacob_T[:, :, i].mm(self.delta_T[:, i].reshape([-1, 1])).squeeze()
            self.Delta_T_S[:, i] = self.jacob_S[:, :, i].mm(self.delta_T[:, i].reshape([-1, 1])).squeeze()
        self.lambda_S_max = self.lambda_S.max().item()
        self.lambda_T_max = self.lambda_T.max().item()
        #deviation /= self.N
        #self.alpha = self.Delta_T_S.pow(2).sum(dim=0).mean().item() / deviation
        self.jacob_S_D_2_norm = self.jacob_S_2_norm.pow(2).mean().pow(0.5).item()
        self.jacob_T_D_2_norm = self.jacob_T_2_norm.pow(2).mean().pow(0.5).item()

    def compute_alpha(self):
        self.alpha_T_S = torch.sqrt(self.Delta_T_S.pow(2).sum(dim=0)) / self.jacob_S_2_norm
        self.alpha_S_T = torch.sqrt(self.Delta_S_T.pow(2).sum(dim=0)) / self.jacob_T_2_norm
        self.alpha_T_S_mean = self.alpha_T_S.mean().item()
        self.alpha_S_T_mean = self.alpha_S_T.mean().item()

    def compute_gamma(self):
        Y_S_T = 0
        Y_T_S = 0
        Y_S_T_combined = 0
        Y_T_S_combined = 0
        for i in range(self.N):
            Y_S_T += torch.outer(unit_vector(self.Delta_S_S[:, i]), unit_vector(self.Delta_S_T[:, i]))
            Y_T_S += torch.outer(unit_vector(self.Delta_T_T[:, i]), unit_vector(self.Delta_T_S[:, i]))
            Y_S_T_combined += self.alpha_S_T[i] * torch.outer(unit_vector(self.Delta_S_S[:, i]), unit_vector(self.Delta_S_T[:, i]))
            Y_T_S_combined += self.alpha_T_S[i] * torch.outer(unit_vector(self.Delta_T_T[:, i]), unit_vector(self.Delta_T_S[:, i]))
        Y_S_T /= self.N
        Y_T_S /= self.N
        Y_S_T_combined /= self.N
        Y_T_S_combined /= self.N
        self.gamma_S_T = Y_S_T.norm().item()
        self.gamma_T_S = Y_T_S.norm().item()
        self.alpha_gamma_combined_S_T = Y_S_T_combined.norm().item()
        self.alpha_gamma_combined_T_S = Y_T_S_combined.norm().item()

    def compute_gradient_loss(self):
        P = 0
        S = 0
        for i in range(self.N):
            P += self.jacob_T[:, :, i].mm(self.jacob_S[:, :, i].T)
            S += self.jacob_S[:, :, i].mm(self.jacob_S[:, :, i].T)
        P /= self.N
        S /= self.N
        self.A = P.mm(S.pinverse())
        loss = 0.0
        for i in range(self.N):
            loss += (self.jacob_T[:, :, i] - self.A.mm(self.jacob_S[:, :, i])).norm().item() ** 2
        self.gradient_loss = (loss / self.N) ** 0.5



    def compute_tau1(self):
        self.tau1 = (self.delta_S * self.delta_T).sum(dim=0).pow(2)
        self.tau1_mean = self.tau1.mean().item()

    def compute_tau2(self):
        # first we the matrix A. Denote A = proj(BC^{pinv}, r).
        # B = torch.zeros([self.d, self.m])
        # C = torch.zeros([self.m, self.m])
        B = torch.zeros([self.d, self.d])
        C = torch.zeros([self.d, self.d])
        Delta_T_S_norm = self.Delta_T_S.pow(2).sum(dim=0).mean().item()
        Delta_S_S_norm = self.Delta_S_S.pow(2).sum(dim=0).mean().item()
        r = (Delta_T_S_norm / Delta_S_S_norm) ** 0.5
        for i in range(self.N):
            B += self.Delta_T_S[:, i].reshape([-1, 1]).mm(self.Delta_S_S[:, i].reshape([1, -1]))
            C += self.Delta_S_S[:, i].reshape([-1, 1]).mm(self.Delta_S_S[:, i].reshape([1, -1]))
        B /= self.N
        C /= self.N
        self.A = B.mm(C.pinverse())
        # projection
        r_A = matrix_2_norm(self.A)
        if r_A > r:
            self.A *= r / r_A

        # compute tau2
        self.tau2 = ((2 * self.Delta_T_S - self.A.mm(self.Delta_S_S)) * self.A.mm(self.Delta_S_S)).sum(
            dim=0).mean() / Delta_T_S_norm
        self.tau2 = self.tau2.item()

    def compute_theorem_2(self):
        # compute LHS in theorem 2
        diff_norm_square = 0
        for i in range(self.N):
            diff_jacob = self.jacob_T[:, :, i] - self.A.mm(self.jacob_S[:, :, i])
            diff_norm_square += matrix_2_norm(diff_jacob) ** 2
        self.theorem_2_LHS = diff_norm_square / self.N

        # compute RHS in theorem 2 (the version without Lipschitz assumption)
        upper_bound_accumulate = 0
        for i in range(self.N):
            # denote upper_bound = a + b
            a = (1 - self.tau1[i]*self.tau2 + (1 - self.tau1[i]) * (1 - self.tau2) * self.lambda_T[i] ** 2) * self.jacob_T_2_norm[i] ** 2
            b = (self.lambda_T[i] ** 2 + self.lambda_S[i] ** 2) * self.jacob_S_2_norm[i] ** 2 / self.jacob_S_D_2_norm ** 2 * self.jacob_T_D_2_norm ** 2
            upper_bound_accumulate += 5 * (a + b)
        self.theorem_2_upper_bound = upper_bound_accumulate.item() / self.N

    def compute_beta(self):
        sigmoid_beta = 1 / 6 / 3 ** 0.5
        W1_S_norm = matrix_2_norm(self.f_S.W1)
        W2_S_norm = matrix_2_norm(self.f_S.W2)
        W1_T_norm = matrix_2_norm(self.f_T.W1)
        W2_T_norm = matrix_2_norm(self.f_T.W2)
        beta_T = sigmoid_beta * W1_T_norm ** 2 * W2_T_norm
        beta_S = sigmoid_beta * W1_S_norm ** 2 * W2_S_norm
        if beta_S > beta_T:
            self.beta = beta_S
        else:
            self.beta = beta_T


def matrix_2_norm(A):
    _, s, _ = torch.svd(A)
    sorted_s, _ = s.abs().sort(descending=True)
    return sorted_s[0].item()

def unit_vector(v: torch.Tensor):
    return v/v.norm()

