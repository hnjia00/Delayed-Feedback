import random
import numpy as np
import math

np.random.seed(1)
random.seed(1)

M = 15
B = 3000
d = 50
w_cut = 1.1
A = np.arange(0,M,1).tolist()
data_size = 10000
data_buffer_counter = 0

criteo_recent_size = 75021
criteo_all_size = 1278556
N = int(criteo_all_size / B)

class Data_config():
    def __init__(self):
        self.time_interval = 0.0014
        self.T = self.time_interval * B
        self.W_C = {}
        self.W_D = {}
        self.CAMPAIGN1 = ['7227c706', '78dcd87f', '431c274d', 'a86b6804', 'c5a9a3c2']
        self.CAMPAIGN = ['7227c706', '78dcd87f', '431c274d', 'a86b6804', 'c5a9a3c2',
                    '12420a1e', '66c8c82f', '5d87968e', '7fde5a70', '6855ef53',
                    '41707ecb', '8379baa1', '95667a0f', 'ad3508b1', '93ec533b'
                    ]
        self.CLICK_prob = np.arange(0.3, 0.5, 0.2/M)
        # self.CLICK_prob = [0.3, 0.35, 0.4, 0.45, 0.5]
        self.S = None

    def set_W(self, W_C, W_D):
        self.W_C = W_C
        self.W_D = W_D

    def set_S(self, S):
        self.S = S

class UCB_config():
    def __init__(self):
        self.miu_ucb = 1.0
        self.Theta = []
        self.Fai = []
        self.Fai_inv = []
        self.B_aj = []
        for i in range(M):
            theta = np.zeros((d, 1))
            self.Theta.append(theta)
            fai = np.ones((d, d))  # 改成单位阵
            self.Fai.append(fai)
            b = np.zeros((d, 1))
            self.B_aj.append(b)

class EXP3_config():
    def __init__(self):
        self.Theta = []
        for i in range(M):
            theta = np.zeros((d, 1))
            self.Theta.append(theta)
        self.P = np.ones((M, 1))
        self.Q = np.ones((M, 1))
        self.delta = 0.1
        self.eta = (2 * (1 - self.delta) * math.log(M) / (M * N * B)) ** 0.5
        self.alpha = 0.0005

class EXP3S_config():
    def __init__(self):
        self.Theta = []
        self.Theta_capital = []
        for i in range(M):
            theta = np.zeros((d, 1))
            self.Theta.append(theta)
            self.Theta_capital.append([])
        self.P = np.ones((M, 1))
        self.Q = np.ones((M, 1))
        self.delta = 0.1
        self.eta = (2 * (1 - self.delta) * math.log(M) / (M * N * B)) ** 0.5
        self.alpha = 0.0005

class EXP3S1_config():
    def __init__(self):
        self.Theta = []
        self.Theta_capital = []
        self.Fai = []
        self.B_aj = []
        for i in range(M):
            theta = np.zeros((d, 1))
            self.Theta.append(theta)
            fai = np.ones((d, d))  # 改成单位阵
            self.Fai.append(fai)
            b = np.zeros((d, 1))
            self.B_aj.append(b)
            self.Theta_capital.append([])
        self.P = np.ones((M, 1))
        self.Q = np.ones((M, 1))
        self.delta = 0.001
        self.eta = (2 * (1 - self.delta) * math.log(M) / (M * N * B)) ** 0.5

class DFM_config():
    def __init__(self):
        self.W_C = []
        self.W_D = []
        self.continuous = 1
        for i in range(M):
            self.W_C.append(np.zeros((d, 1)))
            self.W_D.append(np.zeros((d, 1)))


# {a,s,r,e,y}
# 加载criteo数据
def load_criteo_data(size=M):
    _D = Data_config()
    # 加载w_c和w_d
    W_C = {}
    W_D = {}
    for index in range(size):
        c = _D.CAMPAIGN[index]

        filename_c = './coef/w_c/' + c + '.txt'
        f = open(filename_c, 'r')
        w_c = f.readline().strip('\n').strip('[').strip(']').split(', ')
        w_c = [eval(i) for i in w_c]
        w_c = np.array(w_c) + np.random.normal(0, 0.01, (d,))

        filename_d = './coef/w_d/' + c + '.txt'
        f = open(filename_d, 'r')
        w_d = f.readline().strip('\n').strip('[').strip(']').split(', ')
        w_d = [eval(i) for i in w_d]
        w_d = np.array(w_d) + np.random.normal(0, 10, (d,))

        W_C[c] = w_c
        W_D[c] = w_d

    # dataset = np.loadtxt("recent_criteo_dataset.txt", dtype=float, delimiter=',')
    dataset = np.loadtxt("all_criteo_dataset.txt", dtype=float, delimiter=',')
    dataset_list = []
    for i in range(len(dataset)):
        a = dataset[i][0]
        s = dataset[i][1:51]
        r = dataset[i][51]
        e = dataset[i][52]
        y = dataset[i][53]
        dataset_list.append([a, s.tolist(), r, e, y])

    # recent_state: (75021,50)
    # all_state: (1278556, 50)
    all_state = np.loadtxt("./data_top_pca/all_state_random.txt", dtype=float, delimiter=',')
    # all_state = np.loadtxt("./data_top_pca/recent_state_random.txt", dtype=float, delimiter=',')

    _D.set_S(all_state)
    _D.set_W(W_C, W_D)
    return np.array(dataset_list), _D
