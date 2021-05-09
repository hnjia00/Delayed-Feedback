import matplotlib.pyplot as plt
import numpy as np
import criteo_data.config as config

config.B = 12000
fig, ax = plt.subplots()
font = {'size': 13}
# plt.title('Criteo Data recent#5arms')
plt.xlabel("N", font)
plt.ylabel("Average Reward", font)

f2 = open('./result/final_result/all/CSBDF','r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    reward22.append(sum/config.B/(i+1))
plt.plot(np.arange(0,len(reward22),1), reward22, label='CBDF', linewidth=1)
reward_scatter = []
for j in range(22):
    reward_scatter.append(reward22[j*5])
plt.scatter(np.arange(0,110,5), reward_scatter, marker='^')
print(reward22)

f2 = open('./result/final_result/all/SBUCB','r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    reward22.append(sum/config.B/(i+1))
plt.plot(np.arange(0,len(reward22),1),reward22, label='SBUCB', linewidth=1)
reward_scatter = []
for j in range(22):
    reward_scatter.append(reward22[j*5])
plt.scatter(np.arange(0,110,5), reward_scatter, marker='o')
print(reward22)

f2 = open('./result/final_result/all/DFMS','r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    reward22.append(sum/config.B/(i+1))
plt.plot(np.arange(0,len(reward22),1), reward22, label='DFM-S', linewidth=1)
reward_scatter = []
for j in range(22):
    reward_scatter.append(reward22[j*5])
plt.scatter(np.arange(0,110,5), reward_scatter, marker='s')
print(reward22)

f2 = open('./result/final_result/all/EXP3','r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    reward22.append(sum/config.B/(i+1))
plt.plot(np.arange(0,len(reward22),1), reward22, label='EXP3-B', linewidth=1)
reward_scatter = []
for j in range(22):
    reward_scatter.append(reward22[j*5])
plt.scatter(np.arange(0,110,5), reward_scatter, marker='p')
print(reward22)

reward2 = [2370.694144933677, 2402.861778714495, 2430.223509202571, 2406.081452513942, 2403.211860036833, 2452.1220045371256, 2558.2041390592203, 2428.0649813509453, 2499.954245145057, 2480.2822308163563, 2496.2399264854994, 2410.191553915628, 2401.1537857289263, 2487.4223183739823, 2411.2013867267315, 2546.2168526467362, 2367.1054581097537, 2424.104832713784, 2547.2859117707394, 2409.1428921365978, 2474.013003327885, 2476.0934077288557, 2506.422318373939, 2427.1594058388882, 2539.4292016890668, 2471.234705762187, 2389.9965367784725, 2378.216917416039, 2534.2367707693875, 2517.2568807339067, 2390.8447889406107, 2505.3616501516317, 2505.3895460048234, 2494.2196236449417, 2427.1470672977666, 2535.2787711088577, 2452.1899917965334, 2463.154177293304, 2541.290323170733, 2512.2214271110797, 2427.075000000146, 2485.25401264796, 2544.371052312612, 2494.2350684372695, 2543.4595265797384, 2389.093767307324, 2457.068009527866, 2476.041336376721, 2399.9873117830657, 2421.215064456721, 2563.2454239705467, 2503.2594780835916, 2465.962481203145, 2469.2095393120085, 2543.3503378514924, 2450.1145356994675, 2440.244099632477, 2475.046782825174, 2487.2333980384356, 2571.339330230678, 2377.204890526047, 2417.131547496356, 2429.1049669558356, 2464.0876095946724, 2394.967269688769, 2483.0106634774866, 2372.9992501562056, 2491.1606514590676, 2427.122313879125, 2402.9249245283245, 2427.2450704224357, 2411.2182654940652, 2471.199260245825, 2465.1540045248075, 2451.1899979491463, 2527.270291183051, 2463.182068541022, 2554.2676205983844, 2466.1640937114116, 2537.1717948718706, 2441.0059254631283, 2544.0384089496542, 2475.062423865837, 2499.192875589194, 2347.793878762248, 2397.8621159114387, 2433.7918559321765, 2402.9123483736057, 2397.9199916107427, 2416.182765695614, 2459.921944677351, 2455.1521785641107, 2460.2599673735676, 2431.119948432288, 2541.287680422841, 2451.0251673249477, 2489.248512244954, 2414.069701739831, 2424.0799668874797, 2452.350617283868, 2433.2799348665903, 2493.2249233284233, 2460.177179223843, 2455.92819438622, 2398.8873101998106, 2480.362459049673]
sum = 0
reward22 = []
for i in range(len(reward2)):
    sum += reward2[i]
    # reward11.append(reward1[i]/10000)
    reward22.append(sum/config.B/(i+1))
plt.plot(np.arange(0,len(reward22),1), reward22, label='SBUCB-D', linewidth=1)
reward_scatter = []
for j in range(22):
    reward_scatter.append(reward22[j*5])
plt.scatter(np.arange(0,110,5), reward_scatter, marker='^')
print(reward22)

plt.legend()

plt.show()

fig.savefig('result/final_result/all/criteo_all_result_0205.pdf',format='pdf')