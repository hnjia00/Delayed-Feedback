import matplotlib.pyplot as plt
import numpy as np
import Artificial_data.config as config
from matplotlib.backends.backend_pdf import PdfPages

N = config.N
fig, ax = plt.subplots()
plt.title('CSB-DF')
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

B = 6000
mu = 5
w = 5
file = 'result/final_result/mu/N40_reward_mu'
# file = 'result/final_result/B/continuous_coef_reward_DFM_B'
fmt = ['-^', '-,', '-o', '-v', '-s', '-p', '-*', '-h', '-+', '-x', '-1', '-2', '-3', '-4']

for ii in range(10):
    f = file + str(w) + ''
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
    if w < 10:
        ax.plot(np.arange(0, N, 1), reward22, fmt[ii], label='μ=0.'+str(w), linewidth=0.5)
    else:
        ax.plot(np.arange(0, N, 1), reward22, fmt[ii], label='μ=' + str(w)[0] + '.' + str(w)[1], linewidth=0.5)

    # ax.plot(np.arange(0, N, 1), reward22, fmt[ii], label='B=' + str(B), linewidth=0.5)
    # w += 1
    # B += 1000
    w += 1

plt.legend()
plt.show()
fig.savefig('result/final_result/mu/mu_test.pdf',format='pdf')