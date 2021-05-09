import random
import numpy as np
import config

np.random.seed(1)
random.seed(1)

# buffer modification 需要的变量
Beta = []
for a in config.A:
    beta = np.zeros((config.d,1))
    Beta.append(beta)

def UCB(D_i, n, _UCB):
    # print('n', n)
    i = 0
    S = []
    R = []
    N_a = np.zeros((config.M,1))

    # 按动作排序
    D_i = D_i.tolist()
    D_i.sort(key=(lambda x:x[0]))
    D_i = np.array(D_i)

    for a in config.A:
        S_aj = []
        R_aj = []
        while(i < len(D_i) and D_i[i][0] == a):
            S_aj.append(D_i[i][1])
            R_aj.append(D_i[i][2])
            N_a[a] += 1
            i = i+1

        S.append(np.array(S_aj))
        R.append(np.array(R_aj).reshape((-1,1)))

    for a in config.A:
        _UCB.Fai[a] += np.dot(S[a].T, S[a])
        _UCB.B_aj[a] += np.dot(S[a].T, R[a])
        _UCB.Theta[a] = np.dot(np.linalg.inv(_UCB.Fai[a]), _UCB.B_aj[a])




