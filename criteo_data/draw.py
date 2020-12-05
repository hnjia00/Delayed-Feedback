import matplotlib.pyplot as plt
import numpy as np
import criteo_data.config as config

config.B = 10000
fig, ax = plt.subplots()
font = {'size': 13}
plt.title('Criteo Data recent#5arms')
plt.xlabel("N", font)
plt.ylabel("Average Reward", font)

f2 = open('./result/final_result/recent/B/UCB_all_reward_B10000','r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    reward22.append(sum/config.B/(i+1))
plt.plot(np.arange(0,len(reward22),1), reward22, '-.' ,label='CSB-DF', linewidth=2)
print(reward22)

f2 = open('./result/final_result/recent/B/UCB_all_reward_B10000no','r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    reward22.append(sum/config.B/(i+1))
plt.plot(np.arange(0,len(reward22),1),reward22, '--', label='SBUCB', linewidth=2)
print(reward22)

f2 = open('./result/final_result/recent/B/continuous_coef_reward_DFM_B10000','r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    reward22.append(sum/config.B/(i+1))
plt.plot(np.arange(0,len(reward22),1), reward22, ':' ,label='DFM-S', linewidth=2)
print(reward22)

plt.legend()

plt.show()

# fig.savefig('result/final_result/all/result.pdf',format='pdf')