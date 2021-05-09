import matplotlib.pyplot as plt
import numpy as np
import Artificial_data.config as config

N = config.N
config.B = 10000
fig, ax = plt.subplots()
font = {'size': 13}
# plt.title('Artificial Data')
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
plt.plot(np.arange(0,N,1), reward11, '-.^', label='CBDF', linewidth=1)

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

f1 = open('result/final_result/B/EXP3','r')
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
plt.plot(np.arange(0,N,1), reward11, '-p', label='EXP3-B', linewidth=1)

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

reward1 = [1550.1773438927412, 1488.859126661196, 1565.5357750683536, 1581.607746722696, 1528.7774712643798, 1615.242910279472, 1670.4229096616298, 1583.1332368243432, 1606.3208235553705, 1637.383566822406, 1586.345184216239, 1547.1642507711704, 1594.3748377258805, 1632.4352946349331, 1587.3376518130933, 1604.430815264588, 1602.5515102732131, 1608.5107260655527, 1572.292829694405, 1591.527749365955, 1576.4499808796063, 1577.347023628934, 1550.2843830888232, 1655.725386079947, 1610.5751218479109, 1612.4432855280738, 1591.5333248892825, 1568.4922531964799, 1538.3187080322632, 1600.5084452974231, 1614.6485163454447, 1567.3067585757362, 1564.460244004981, 1634.6806070826985, 1581.5618126824666, 1589.537754371226, 1608.5937861209704, 1576.5847599337276, 1574.5354590984089, 1574.3908669528887]
sum = 0
reward11 = []
for i in range(len(reward1)):
    sum += reward1[i]
    # reward11.append(reward1[i]/10000)
    reward11.append(sum/config.B/(i+1))
plt.plot(np.arange(0,N,1), reward11, '-^', label='SBUCB-D', linewidth=1)


plt.legend()

plt.show()

fig.savefig('result/final_result/Artificial_result_0205.pdf',format='pdf')
