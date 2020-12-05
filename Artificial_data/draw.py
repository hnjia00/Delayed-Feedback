import matplotlib.pyplot as plt
import numpy as np
import Artificial_data.config as config

N = config.N
config.B = 10000
fig, ax = plt.subplots()
font = {'size': 13}
plt.title('Artificial Data')
plt.xlabel("N", font)
plt.ylabel("Average Reward", font)

f1 = open('result/final_result/w/N40_reward_w18','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])
sum = 0
reward11 = []
for i in range(len(reward1)):
    sum += reward1[i]
    # reward11.append(reward1[i]/10000)
    reward11.append(sum/config.B/(i+1))
plt.plot(np.arange(0,N,1), reward11, '-.^', label='BDF-T', linewidth=1)

f1 = open('result/final_result/mu/N40_reward_mu11','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])
sum = 0
reward11 = []
for i in range(len(reward1)):
    sum += reward1[i]
    # reward11.append(reward1[i]/10000)
    reward11.append(sum/config.B/(i+1))
plt.plot(np.arange(0,N,1), reward11, '--o', label='SBUCB', linewidth=1)

f2 = open('result/final_result/B/continuous_coef_reward_DFM_B10000','r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    # reward11.append(reward1[i]/10000)
    reward22.append(sum/config.B/(i+1))
plt.plot(np.arange(0,N,1), reward22, ':s', label='DFM-S', linewidth=1)

f1 = open('result/final_result/w/N40_reward_replay11_w18','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])
sum = 0
reward11 = []
for i in range(len(reward1)):
    sum += reward1[i]
    # reward11.append(reward1[i]/10000)
    reward11.append(sum/config.B/(i+1))
plt.plot(np.arange(0,N,1), reward11, '-p', label='BDF-R', linewidth=1)

# f2 = open('result/DFM/buffer_modify/continuous_coef_reward_DFM_B6000','r')
# line = f2.readline().strip('\n').strip('[').strip(']')
# reward2 = line.split(', ')
# for i in range(len(reward2)):
#     reward2[i] = eval(reward2[i])
# sum = 0
# reward22 = []
# for i in range(len(reward2)):
#     sum += reward2[i]
#     # reward11.append(reward1[i]/10000)
#     reward22.append(sum/config.B/(i+1))
# plt.plot(np.arange(0,N,1), reward22, label='reward_DFM_continuous_coef')


plt.legend()

plt.show()

fig.savefig('result/final_result/result1.pdf',format='pdf')
