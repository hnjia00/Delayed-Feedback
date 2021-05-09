import random
import numpy as np
import criteo_data.config as config

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
        t1 = np.dot(S[a], _EXP3S.Theta[a])
        t2 = t1 - R[a]
        t3 = np.dot(S[a].T, t2)
        _EXP3S.Theta[a] = _EXP3S.Theta[a] - _EXP3S.alpha * t3
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
