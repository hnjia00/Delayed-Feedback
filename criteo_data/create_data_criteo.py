import random
import numpy as np
import criteo_data.config as config
import math

np.random.seed(1)
random.seed(1)

data_size = 10000
A = np.arange(0,15,1).tolist()

def load_criteo_data(size=15):
    CAMPAIGN = ['7227c706', '78dcd87f', '431c274d', 'a86b6804', 'c5a9a3c2'
                ,'12420a1e', '66c8c82f', '5d87968e', '7fde5a70', '6855ef53'
                ,'41707ecb', '8379baa1', '95667a0f', 'ad3508b1', '93ec533b'
                ]
    CLICK_prob = np.arange(0.3, 0.5, 0.2/config.M).tolist()
    T = 14
    time_interval = T/data_size

    # 加载w_c和w_d
    W_C = {}
    W_D = {}
    for index in range(size):
        c = CAMPAIGN[index]

        filename_c = './coef/w_c/' + c + '.txt'
        f = open(filename_c, 'r')
        w_c = f.readline().strip('\n').strip('[').strip(']').split(', ')
        w_c = [eval(i) for i in w_c]
        w_c = np.array(w_c)

        filename_d = './coef/w_d/' + c + '.txt'
        f = open(filename_d, 'r')
        w_d = f.readline().strip('\n').strip('[').strip(']').split(', ')
        w_d = [eval(i) for i in w_d]
        w_d = np.array(w_d)

        W_C[c] = w_c
        W_D[c] = w_d

    # 加载pca数据
    for index in range(size):
        c = CAMPAIGN[index]
        filename = './data_top_pca/' + c + '.txt'
        if index == 0:
            data = np.loadtxt(filename, dtype=float, delimiter=',')
        else:
            _ = np.loadtxt(filename, dtype=float, delimiter=',')
            data = np.append(data, _, axis=0)

    N_click = 0
    N_conversion = 0
    N_y = 0
    dataset = []
    label = []
    for i in range(data_size):
        s = data[i, :]
        a = random.sample(A, 1)[0]

        c = int(np.random.choice([0, 1], p=[1-CLICK_prob[a], CLICK_prob[a]]))

        if c == 1:
            N_click += 1
            campaign_i = CAMPAIGN[a]
            w_c_i = W_C[campaign_i]
            w_d_i = W_D[campaign_i]

            cvr = 1 / (1 + np.exp(-np.dot(w_c_i, s)))
            v = 0
            if random.random() <= cvr:
                v = 1
                N_conversion += 1
            lamda_s = np.exp(np.dot(w_d_i,s))
            gamma = random.expovariate(lamda_s) / 86400

            d = 0
            if i * time_interval + gamma <= T:
                d = 1
            y = d * v

        else:
            v = 0
            d = 0
            gamma = 0
            y = 0

        dataset.append([s,a,c,v,d,gamma])
        label.append(y)

    data_buffer = []
    lamda = 0.01 * N_conversion / N_click
    for i in range(data_size):
        s = dataset[i][0]
        a = dataset[i][1]

        # 计算r
        c = dataset[i][2]
        # v = dataset[i][3]
        gamma = dataset[i][5]
        y = label[i]

        r_head = 0
        if c == 1:
            r_head = 1

        r_tilde = 0
        if y == 1:
            r_tilde = 1

        e_i = 0
        if c == 1:
            if y == 1:
                N_y += 1
                e_i = gamma
            else:
                e_i = T - time_interval * i

        # r = lamda * (r_head - e_i/T) + (1 - lamda) * r_tilde

        r = lamda * r_head + (1 - lamda) * r_tilde

        if type(r) is not float:
            r = float(r)

        if type(e_i) is not float:
            e_i = float(e_i)

        # a: [0], s: [1:50], r: [51], e_i: [52], y: [53]
        t1 = np.append(np.array([a]), s, axis=0)
        data_buffer.append(np.append(t1, np.array([r, e_i, y]), axis=0))

    print(N_click)
    print(N_conversion)
    print(N_y)

    data_buffer = np.array(data_buffer)
    np.savetxt("all_criteo_dataset.txt", data_buffer, fmt='%f', delimiter=',')

    # return data_buffer


def concat_pca():
    f1 = np.loadtxt("./data_top_pca/7227c706.txt", dtype=float, delimiter=',')
    f2 = np.loadtxt("./data_top_pca/78dcd87f.txt", dtype=float, delimiter=',')
    f = np.append(f1[10000:], f2, axis=0)
    f2 = np.loadtxt("./data_top_pca/431c274d.txt", dtype=float, delimiter=',')
    f = np.append(f, f2, axis=0)
    f2 = np.loadtxt("./data_top_pca/a86b6804.txt", dtype=float, delimiter=',')
    f = np.append(f, f2, axis=0)
    f2 = np.loadtxt("./data_top_pca/c5a9a3c2.txt", dtype=float, delimiter=',')
    f = np.append(f, f2, axis=0)
    np.savetxt("./data_top_pca/all_state.txt", f,  fmt='%f', delimiter=',')

def random_concat_pca():
    print("begin")
    CAMPAIGN = ['7227c706', '78dcd87f', '431c274d', 'a86b6804', 'c5a9a3c2'
                ,'12420a1e', '66c8c82f', '5d87968e', '7fde5a70', '6855ef53'
                ,'41707ecb', '8379baa1', '95667a0f', 'ad3508b1', '93ec533b'
                ]
    F = []
    L = []
    f1 = np.loadtxt("./data_top_pca/7227c706.txt", dtype=float, delimiter=',')[10000:]
    F.append(f1)
    L.append(f1.shape[0])
    # f2 = np.loadtxt("./data_top5_pca/78dcd87f.txt", dtype=float, delimiter=',')
    # len2 = f2.shape[0]
    # f3 = np.loadtxt("./data_top5_pca/431c274d.txt", dtype=float, delimiter=',')
    # len3 = f3.shape[0]
    # f4 = np.loadtxt("./data_top5_pca/a86b6804.txt", dtype=float, delimiter=',')
    # len4 = f4.shape[0]
    # f5 = np.loadtxt("./data_top5_pca/c5a9a3c2.txt", dtype=float, delimiter=',')
    # len5 = f5.shape[0]
    print("read data")
    for campaign in CAMPAIGN[1:]:
        file = "./data_top_pca/" + campaign + ".txt"
        f = np.loadtxt(file, dtype=float, delimiter=',')
        l = f.shape[0]
        F.append(f)
        L.append(l)

    print("finish read data")
    cnt = np.zeros((len(CAMPAIGN),1))
    f = np.reshape(f1[0], (1, 50))
    cnt[0] += 1
    flag = np.ones((len(CAMPAIGN)))
    while np.sum(flag) != 0:
        i = random.sample(A, 1)[0]
        if cnt[i] % 2000 == 0:
            print(cnt.tolist())
        if cnt[i] == L[i]:
            A.remove(i)
            flag[i] = 0
            continue

        t = np.reshape(F[i][int(cnt[i])], (1, 50))
        f = np.append(f, t, axis=0)
        cnt[i] += 1

    np.savetxt("./data_top_pca/all_state_random.txt", f,  fmt='%f', delimiter=',')

# random_concat_pca()
load_criteo_data()