import random
import numpy as np
import math

np.random.seed(1)
random.seed(1)

M = 5
N = 40
B = 10000
d = 10
A = np.arange(0,M,1).tolist()
data_size = 12000
data_buffer_counter = 0
w_cut = 1.8

class Data_config():
    def __init__(self):
        self.sigma_c = 0.01
        self.sigma_d = 0.01
        self.time_interval = 0.001  # 调小
        self.T = self.time_interval * B
        self.miu_c = 0
        self.miu_d = 0
        self.w_c = []
        self.w_d = []
        for i in range(M):
            w_c_i = np.random.normal(self.miu_c, ((i + 1) * self.sigma_c) ** 2, (1, d))
            w_d_i = np.random.normal(self.miu_d, ((i + 1) * self.sigma_d) ** 2, (1, d))
            self.miu_c -= 0.2
            self.miu_d += 0.2
            self.w_c.append(w_c_i)
            self.w_d.append(w_d_i)

        # Criteo
        # self.time_interval = 0.0014
        # self.T = self.time_interval * B
        # self.W_C = {}
        # self.W_D = {}
        # self.CAMPAIGN = ['7227c706', '78dcd87f', '431c274d', 'a86b6804', 'c5a9a3c2']
        # self.CAMPAIGN1 = ['7227c706', '78dcd87f', '431c274d', 'a86b6804', 'c5a9a3c2',
        #             '12420a1e', '66c8c82f', '5d87968e', '7fde5a70', '6855ef53',
        #             '41707ecb', '8379baa1', '95667a0f', 'ad3508b1', '93ec533b'
        #             ]
        # self.S = None
        # self.CLICK_prob = np.arange(0.3, 0.5, 0.2/M)
        self.CLICK_prob = [0.3, 0.35, 0.4, 0.45, 0.5]

    def set_W(self, W_C, W_D):
        self.W_C = W_C
        self.W_D = W_D

    def set_S(self, S):
        self.S = S

class UCB_config():
    def __init__(self):
        self.miu_ucb = 2
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

# 加载人工静态数据
# {a,s,r,e,y}
def load_data():
    data_buffer = []
    f = open('data1.txt', 'r')
    lines = f.readlines()
    for line in lines:
        l1 = line.strip('[').strip(']').strip('\n')
        l2 = l1.split('|')

        # 切割动作a、奖励r
        l2_1 = l2[1].strip('\', ').strip(']').split(', ')
        a = int(eval(l2_1[0]))
        r = eval(l2_1[1])
        e = eval(l2_1[2])
        y = int(eval(l2_1[3]))

        # 修正延迟
        if y == 0 and e == 0:
            e = data_size * 0.001

        # 切割状态s
        l2_0 = l2[0].strip('], \'').split('], [')
        s = []
        for item in l2_0:
            s.append(eval(item))

        data_buffer.append([a,s,r,e,y])

    return np.array(data_buffer)



