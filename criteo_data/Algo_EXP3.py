import random
import numpy as np
import criteo_data.config as config

np.random.seed(1)
random.seed(1)

def EXP3(pai, D_i, n, _EXP3):
    i = 0
    S = []
    R = []
    N_a = np.zeros((config.M, 1))

    # 按动作排序
    D_i = D_i.tolist()
    D_i.sort(key=(lambda x: x[0]))
    D_i = np.array(D_i)

    for a in config.A:
        S_aj = []
        R_aj = []
        while (i < len(D_i) and D_i[i][0] == a):
            S_aj.append(D_i[i][1])
            R_aj.append(D_i[i][2])
            N_a[a] += 1
            i = i + 1

        S.append(np.array(S_aj))
        R.append(np.array(R_aj).reshape((-1, 1)))

    for a in config.A:
        t1 = np.dot(S[a], _EXP3.Theta[a])
        t2 = t1 - R[a]
        t3 = np.dot(S[a].T, t2)
        _EXP3.Theta[a] = _EXP3.Theta[a] - _EXP3.alpha * t3
        # Theta[a] = Theta[a] - alpha*(np.dot(S[a].T,(np.dot(S[a],Theta[a]) - R[a])))
        r_head = np.sum(np.dot(S[a], _EXP3.Theta[a])) / N_a[a]
        # r_head = np.sum(np.dot(S[a], Theta[a]))
        _EXP3.P[a] = _EXP3.P[a] * np.exp(_EXP3.eta * r_head / pai[a])

    p_sum = np.sum(_EXP3.P)

    for a in config.A:
        _EXP3.Q[a] = _EXP3.P[a] / p_sum

    pai = (1 - _EXP3.delta) * _EXP3.Q + _EXP3.delta / config.M

    return pai


# modify
# [[0.20054344]
#  [0.20503924]
#  [0.20111699]
#  [0.19775612]
#  [0.19554422]]

# no modify
# [[0.20178312]
#  [0.20115944]
#  [0.20009012]
#  [0.19877942]
#  [0.19818791]]