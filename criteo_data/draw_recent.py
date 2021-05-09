import matplotlib.pyplot as plt
import numpy as np
import criteo_data.config as config

config.B = 3000
fig, ax = plt.subplots()
font = {'size': 13}
# plt.title('Criteo Data recent#5arms')
plt.xlabel("N", font)
plt.ylabel("Average Reward", font)

f2 = open('./result/test/UCB_all_reward_B3000','r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    reward22.append(sum/config.B/(i+1))
# plt.plot(np.arange(0,len(reward22),1), reward22, '-.' ,label='CSB-DF', linewidth=2)
plt.plot(np.arange(0,len(reward22),1), reward22, '-^' ,label='CBDF', linewidth=1)
print(reward22)

f2 = open('./result/test/UCB_all_reward_B3000no', 'r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    reward22.append(sum/config.B/(i+1))
# plt.plot(np.arange(0,len(reward22),1),reward22, '--', label='SBUCB', linewidth=2)
plt.plot(np.arange(0,len(reward22),1),reward22, '-o', label='SBUCB', linewidth=1)
print(reward22)

f2 = open('./result/test/replay_continuous_coef_reward_DFM_B3000', 'r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    reward22.append(sum/config.B/(i+1))
# plt.plot(np.arange(0,len(reward22),1), reward22, ':', label='DFM-S', linewidth=2)
plt.plot(np.arange(0,len(reward22),1), reward22, '-s', label='DFM-S', linewidth=1)
print(reward22)

f2 = open('./result/final_result/recent/B/EXP3', 'r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    reward22.append(sum/config.B/(i+1))
# plt.plot(np.arange(0,len(reward22),1), reward22, ':', label='DFM-S', linewidth=2)
plt.plot(np.arange(0,len(reward22),1), reward22, '-p', label='EXP3-B', linewidth=1)
print(reward22)

reward2 = [569.8720104438613, 539.7952767857096, 603.9254483347489, 580.9147855917608, 607.9081545064397, 566.8883044982756, 551.8250928382017, 572.9254483347602, 571.8018554861704, 592.982145850795, 598.0386101973648, 594.9524809483551, 572.9395921835211, 596.8638360941518, 562.8754126846218, 612.9678451178486, 594.0540637775963, 585.978582214766, 573.8968766177752, 568.859685314689, 586.9496949152599, 620.0949676898207, 603.967026116255, 634.9775397489492, 594.8987586206936]
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    # reward11.append(reward1[i]/10000)
    reward22.append(sum/config.B/(i+1))
plt.plot(np.arange(0,len(reward22),1), reward22, '-^', label='SBUCB-D', linewidth=1)

plt.legend()

plt.show()

fig.savefig('result/final_result/recent/criteo_recent_result_0205.pdf',format='pdf')