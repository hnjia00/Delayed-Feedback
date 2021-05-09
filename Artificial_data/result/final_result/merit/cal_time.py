import numpy as np

f = open('./time.txt','r')

lines = f.readlines()

time_dfm = []
time_csbdf = []
time_sbucb = []

i = 1
while i < len(lines):
    time_dfm.append(eval(lines[i])*40)
    time_csbdf.append(eval(lines[i+2])*40)
    time_sbucb.append(eval(lines[i+4])*40)

    i += 6

time_dfm = np.array(time_dfm)
mean_dfm = np.mean(time_dfm)
var_dfm = np.var(time_dfm)
std_dfm = np.sqrt(var_dfm)
print('dfm', mean_dfm, var_dfm, std_dfm)

time_csbdf = np.array(time_csbdf)
mean_csbdf = np.mean(time_csbdf)
var_csbdf = np.var(time_csbdf)
std_csbdf = np.sqrt(var_csbdf)
print('csbdf', mean_csbdf, var_csbdf, std_csbdf)

time_sbucb = np.array(time_sbucb)
mean_sbucb = np.mean(time_sbucb)
var_sbucb = np.var(time_sbucb)
std_sbucb = np.sqrt(var_sbucb)
print('sbucb', mean_sbucb, var_sbucb, std_sbucb)

