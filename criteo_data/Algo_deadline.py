import random
import numpy as np
from tqdm import tqdm
from tqdm import trange
import criteo_data.config as config
import criteo_data.Algo_UCB as Algo_UCB
import criteo_data.Algo_EXP3 as Algo_EXP3
import criteo_data.Algo_EXP3S as Algo_EXP3S
import criteo_data.Algo_EXP3S1 as Algo_EXP3S1
import criteo_data.Algo_modify as Algo_modify
import criteo_data.Algo_DFM as Algo_DFM
import time

np.random.seed(1)
random.seed(1)


def Algo_main(D_n, B, N, _D, _algo=None, mode='no', modify='', xi=0.2):
    pai = np.full((config.M, 1), 1 / config.M)
    # 记录线上累计奖励
    REWARD = []
    C_sum_record = [0]
    V_sum_record = [0]
    for n in tqdm(range(N)):

        C_sum = 0

        # sample_index = np.random.choice(len(D), B)
        # D_n = D[sample_index, :]
        # print('-------' * 20)
        # count_reward(D_n)

        # modify
        if modify != 'no':
            D_n = Algo_modify.modify(D_i=D_n, xi=xi * B, time_interval=_D.time_interval, rho=1 / ((n + 1) * B), n=n)

        if mode == 'UCB':
            Algo_UCB.UCB(D_i=D_n, n=n, _UCB=_algo)
            # 线下计算固定的求逆矩阵
            for a_j in config.A:
                _algo.Fai_inv.append(np.linalg.inv(_algo.Fai[a_j]))
        elif mode == 'EXP3':
            pai = Algo_EXP3.EXP3(pai=pai, D_i=D_n, n=n, _EXP3=_algo)
        elif mode == 'EXP3S':
            Algo_EXP3S.EXP3S_1(D_i=D_n, n=n, _EXP3S=_algo)
            print(pai)
            print('-------' * 20)
        elif mode == 'EXP3S1':
            Algo_EXP3S1.EXP3S_1(D_i=D_n, n=n, _EXP3S=_algo)
            print(pai)
            print('-------' * 20)
        elif mode == 'DFM':
            X, Y, timestamp = Algo_DFM.data_process(D_n)
            for i in range(config.M):
                DFM = Algo_DFM.train_dfm(X[i], Y[i], timestamp[i], _algo.W_C[i], _algo.W_D[i],
                                         continuous=_algo.continuous)
                _algo.W_C[i] = DFM.coef_[:config.d]
                _algo.W_D[i] = DFM.coef_[config.d:]

        N_conv = 0
        N_click = 0
        D_temp = []
        V_record = []  # 辅助线上绘图

        for b in range(B):
            # State: s
            s_Bn_b = _D.S[B * n + b]

            if mode == 'UCB':
                pai = []
                for a_j in config.A:
                    t1 = np.dot(_algo.Theta[a_j].T, s_Bn_b)
                    # t2 = np.dot(s_Bn_b.T, np.linalg.inv(_algo.Fai[a_j]))
                    t2 = np.dot(s_Bn_b.T, _algo.Fai_inv[a_j])
                    t3 = np.sqrt(np.dot(t2, s_Bn_b))
                    pai.append(t1 + _algo.miu_ucb * t3)
                a = np.argmax(pai)
            elif mode == 'EXP3':
                a = np.random.choice(config.A, p=np.reshape(pai, (config.M,)))
            elif mode == 'EXP3S':
                a = np.random.choice(config.A, p=np.reshape(pai, (config.M,)))
                pai = Algo_EXP3S.EXP3S_2(s=s_Bn_b, a=a, n=n, _EXP3S=_algo)
            elif mode == 'EXP3S1':
                a = np.random.choice(config.A, p=np.reshape(pai, (config.M,)))
                pai = Algo_EXP3S1.EXP3S_2(s=s_Bn_b, a=a, n=n, _EXP3S=_algo)
            elif mode == 'DFM':
                cvr_list = [1 / (1 + np.exp(-np.dot(_algo.W_C[i], s_Bn_b))) for i, _ in
                            enumerate(_D.CAMPAIGN)]
                a = cvr_list.index(max(cvr_list))

            c = np.random.choice([0, 1], p=[1 - _D.CLICK_prob[a], _D.CLICK_prob[a]])

            if c == 0:
                cvr = 0
                v = 0
                gamma = _D.T
            else:
                C_sum += 1
                cvr = 1 / (1 + np.exp(-np.dot(_D.W_C[_D.CAMPAIGN[a]], s_Bn_b)))
                v = np.random.choice([0, 1], p=[1 - cvr, cvr])
                lamda_s = np.exp(np.dot(_D.W_D[_D.CAMPAIGN[a]], s_Bn_b))
                gamma = random.expovariate(lamda_s) / 86400

            d = 0
            if b * _D.time_interval + gamma <= _D.T:
                d = 1

            y = d * v

            e_i = 0
            if c == 1:
                if y == 1:
                    e_i = gamma
                else:
                    e_i = _D.T - _D.time_interval * b

            if c == 1:
                N_click += 1
            if v == 1:
                N_conv += 1

            # 线上收集数据用v替换y
            V_record.append(v)
            D_temp.append([a, s_Bn_b.tolist(), c, y, e_i])


        # 根据延迟标签计算奖励
        r_sum = 0
        lamda = 0.01 * N_conv / N_click
        for i in range(B):
            _, _, c, y, _ = D_temp[i]
            v = V_record[i]

            r_head = 0
            if c == 1:
                r_head = 1
            r_wave = 0
            if v == 1:
                r_wave = 1
            r_online = lamda * r_head + (1 - lamda) * r_wave
            r_sum += r_online

            if y == 0:
                r_wave = 0
            r = lamda * r_head + (1 - lamda) * r_wave

            e_i = D_temp[i].pop()
            y = D_temp[i].pop()
            c = D_temp[i].pop()
            D_temp[i].append(r)
            D_temp[i].append(e_i)
            D_temp[i].append(y)

        REWARD.append(r_sum)

        # 存储线上数据
        D_temp = np.array(D_temp)
        # 经验池还没填满
        # if len(D) < config.data_size:
        #     if config.data_buffer_counter + B < config.data_size:
        #         np.append(D, D_temp, axis=0)
        #     else:
        #         np.append(D, D_temp[:config.data_size - config.data_buffer_counter], axis=0)
        #         D[:B + config.data_buffer_counter - config.data_size] = D_temp[
        #                                                                 config.data_size - config.data_buffer_counter:]
        # elif config.data_buffer_counter + B > config.data_size:
        #     D[config.data_buffer_counter:] = D_temp[0:config.data_size - config.data_buffer_counter]
        #     D[:B + config.data_buffer_counter - config.data_size] = D_temp[
        #                                                             config.data_size - config.data_buffer_counter:]
        # else:
        #     D[config.data_buffer_counter:config.data_buffer_counter + B] = D_temp
        # config.data_buffer_counter += B
        # config.data_buffer_counter %= config.data_size

        D_n = D_temp

        # CVR: V==1/C==1
        # CTCVR: V==1/all
        C_sum_record.append(C_sum_record[-1] + C_sum)
        V_sum_record.append(V_sum_record[-1] + sum(V_record))

        # print(b,n)

    return REWARD, C_sum_record, V_sum_record

# 加载静态数据
# {a,s,r,e,y}
if __name__ == '__main__':
    D_init, _D_init = config.load_criteo_data()
    #
    # config.B = 3000
    config.B = 12000
    config.N = int(config.criteo_all_size / config.B)
    # config.N = 1
    mu = 0.8
    w = 1.0

    for ii in range(7):
        R = []
        CVR = []
        CTCVR = []
        for i in range(10):

            mode = 'UCB'
            config.w_cut = w
            Algo_modify.modify_init()
            D = D_init
            _D = _D_init
            _algo = config.UCB_config()
            _algo.miu_ucb = mu
            modify = ''
            reward, c, v = Algo_main(np.array(D), config.B, config.N, _D, _algo, mode, modify=modify, xi=0.2 + ii*0.1)
            # print(reward)
            # print(np.sum(np.array(reward))/config.N, v[-1]/c[-1], v[-1]/(config.N * config.B))

            R.append(np.sum(np.array(reward)) / (config.N * config.B))
            CVR.append(v[-1] / c[-1])
            CTCVR.append(v[-1] / (config.N * config.B))

        # print('xi: ', 0.2 + ii*0.1)
        R = np.array(R)
        CVR = np.array(CVR)
        CTCVR = np.array(CTCVR)
        # print('reward ', np.mean(R), np.std(R))
        # print('CVR ', np.mean(CVR), np.std(CVR))
        # print('CTCVR ', np.mean(CTCVR), np.std(CTCVR))

        print("%d		& %.4f $\pm$ %.4f & %.4f $\pm$ %.4f & %.4f $\pm$ %.4f \\\\"%(100*(0.2+int(ii)*0.1),
                                                                                        np.mean(CVR), np.std(CVR),
                                                                                        np.mean(CTCVR), np.std(CTCVR),
                                                                                        np.mean(R), np.std(R))
              )