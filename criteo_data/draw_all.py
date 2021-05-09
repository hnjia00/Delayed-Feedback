import numpy as np
import criteo_data.config as config
import matplotlib.pyplot as plt


N = config.N
fig, ax = plt.subplots()
# plt.title('CSB-DF')
# plt.title('SBUCB')
# plt.title('DFM-S')
font = {'size': 13}
plt.xlabel("N", font)
plt.ylabel("Average Reward", font)

miu = 5 # miu=0.5/1.4
w = 20
# F = [
#     # './result/UCB/buffer_modify/miu=0.5/recent_reward_w10',
#     # './result/UCB/buffer_modify/miu=0.5/recent_reward_w21',
#     # './result/UCB/buffer_modify/miu=0.5/recent_reward_w28',
#     './result/UCB/buffer_modify/miu=1.4/recent_reward_w10',
#     './result/UCB/buffer_modify/miu=1.4/recent_reward_w21',
#     'result/UCB/buffer_modify/recent_reward_miu7no'
# ]

F = [
    'result/final_result/B/N40_reward_B6000',
    'result/final_result/B/N40_reward_B7000',
    'result/final_result/B/N40_reward_B8000',
    'result/final_result/B/N40_reward_B9000',
    'result/final_result/B/N40_reward_B10000',
    'result/final_result/B/N40_reward_B11000',
    'result/final_result/B/N40_reward_B12000',
    'result/final_result/B/N40_reward_B13000',
]

B = 1000
# mu = 5
# w = 5
# file = 'result/final_result/all/B/UCB_all_reward_B'
file = 'result/test/UCB_all_reward_B'
# file = 'result/final_result/all/mu/UCB_all_reward_mu'
# file = 'result/UCB/buffer_modify/B/recent_reward_B'
# file = 'result/final_result/all/B/continuous_coef_reward_DFM_B'
fmt = ['-,', '-o', '-^', '-v', '-s', '-p', '-*', '-h', '-+', '-x', '-1', '-2', '-3', '-4']
# B = 12000
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['10%N', '20%N', '30%N', '40%N', '50%N', '60%N', '70%N', '80%N', '90%N', 'N'])
for ii in range(6):
    N = int(config.criteo_recent_size/B)
    # f = file + str(round(w)) + ''
    f = file + str(B) + ''
    print(f)
    f2 = open(f, 'r')
    line = f2.readline().strip('\n').strip('[').strip(']')
    reward2 = line.split(', ')
    for i in range(len(reward2)):
        reward2[i] = eval(reward2[i])
    sum = 0
    reward22 = []
    for i in range(len(reward2)):
        sum += reward2[i]
        reward22.append(sum / B / (i + 1))

    reward2 = np.array(reward2)
    reward222 = []
    for j in range(10):
        n = int(N * (j+1) / 10)
        reward222.append(np.sum(reward2[:n])/(n*B))

    ax.plot(np.arange(0, 10, 1), reward222, fmt[ii+1], label='B=' + str(B), linewidth=1)
    # if w < 10:
    #     ax.plot(np.arange(0, N, 1), reward22, '-', label='mu=0.'+str(w), linewidth=1.5)
    # else:
    #     ax.plot(np.arange(0, N, 1), reward22, '-', label='mu=' + str(w)[0] + '.' + str(w)[1], linewidth=1.5)

    # w += 1
    B += 1000
    # mu += 1

plt.legend(loc='lower right')
plt.show()
fig.savefig('result/test/result_B_test_0208.pdf', format='pdf')