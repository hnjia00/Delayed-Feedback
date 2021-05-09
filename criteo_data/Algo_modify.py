import random
import numpy as np
import math
import criteo_data.config as config

np.random.seed(1)
random.seed(1)

# buffer modification 需要的变量
Beta = []

def modify_init():
    global Beta
    Beta = []
    for a in config.A:
        beta = np.zeros((config.d,1))
        Beta.append(beta)


def modify(D_i, xi, time_interval, rho, n):
    Q = []
    for a in config.A:
        Q.append([])
    delete_index = []
    # 修正e_mod
    for i in range(1, int(xi) + 1):
        a, s, r, e, y = D_i[i-1]
        a = int(a)
        if y == 1:
            delete_index.append(i-1)
            D_mod = 0
            if time_interval * i + e < time_interval * xi:
                D_mod = 1
            e_mod = e
            if D_mod == 0:
                e_mod = time_interval * (xi - i)
            # 剔除引发故障 e == 0 的点
            if e_mod < 1e-10:
                continue
            Q[a].append([a, s, r, e_mod, D_mod])

    for a in config.A:
        L = np.zeros((config.d,1))
        # print('n,a',n,a)
        for i in range(len(Q[a])):
            a, s, r, e, D = Q[a][i]
            a = int(a)
            s = np.array(s).reshape((config.d,1))
            lamda = math.exp(np.dot(Beta[a].T, s))
            # print('n,a,i,lamda',n,a,i,lamda)
            L += e * (1 - D - D/(math.exp(lamda * e) - 1)) * lamda * s
            # print('n,a,i,L',n,a,i,L)

        Beta[a] -= rho * L
        # print('n,a,Beta', n, a, Beta)
    # 修正R_mod
    for i in range(len(D_i)):
        # print('n,i',n,i)
        a, s, r, e, y = D_i[i]
        a = int(a)
        if y == 0 or e < 1e-10 :
            continue
        # Q中只包含y=1的项，说明C和V一定等于1
        R_head = 1
        R_tilde = 1
        lamda = math.exp(np.dot(Beta[a].T, s))
        pr = math.exp(-lamda * e)
        w = 1 / (1 - pr)
        # 剔除w
        if w > config.w_cut:
            w = config.w_cut
        R_mod = lamda * R_head + (1 - lamda) * w * R_tilde
        # if R_mod > 0:
        #     R_mod = math.log(R_mod)
        D_i[i][2] = R_mod

    D_i = D_i.tolist()
    for a in config.A:
        for i in range(len(Q[a])):
            D_i.append(Q[a][i])

    return np.array(D_i)