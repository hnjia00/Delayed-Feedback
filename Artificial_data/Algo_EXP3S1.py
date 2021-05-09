import random
import numpy as np
import Artificial_data.config as config

np.random.seed(1)
random.seed(1)

def EXP3S_1(D_i, n, _EXP3S):
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
        _EXP3S.Fai[a] += np.dot(S[a].T, S[a])
        _EXP3S.B_aj[a] += np.dot(S[a].T, R[a])
        _EXP3S.Theta[a] = np.dot(np.linalg.inv(_EXP3S.Fai[a]), _EXP3S.B_aj[a])

        if len(_EXP3S.Theta_capital[a]) == 0:
            _EXP3S.Theta_capital[a] = _EXP3S.Theta[a]
        else:
            _EXP3S.Theta_capital[a] = np.concatenate((_EXP3S.Theta_capital[a], _EXP3S.Theta[a]), axis=1)

def EXP3S_2(s, a, n, _EXP3S):
    t = _EXP3S.Theta_capital[a]
    R_s = np.sum(np.dot(t.T, s))
    _EXP3S.P[a] = np.exp(_EXP3S.eta * R_s)

    p_sum = np.sum(_EXP3S.P)
    for a in config.A:
        _EXP3S.Q[a] = _EXP3S.P[a] / p_sum
    pai = (1 - _EXP3S.delta) * _EXP3S.Q + _EXP3S.delta / config.M

    return pai

# ESP3 modify
# [[0.20054344]
#  [0.20503924]
#  [0.20111699]
#  [0.19775612]
#  [0.19554422]]

# ESP3S modify
# [[0.20096113]
#  [0.19987017]
#  [0.20019284]
#  [0.19896788]
#  [0.20000796]]

# no modify
# [[0.20178312]
#  [0.20115944]
#  [0.20009012]
#  [0.19877942]
#  [0.19818791]]