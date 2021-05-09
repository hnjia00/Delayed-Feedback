import numpy as np
import criteo_data.config as config

f = open('./CVR_CTCVR.txt','r')

lines = f.readlines()

cvr_dfm = []
cvr_csbdf = []
cvr_sbucb = []

ctcvr_dfm = []
ctcvr_csbdf = []
ctcvr_sbucb = []

size = config.criteo_all_size

i = 0
while i < len(lines):
    cvr_dfm.append(eval(lines[i + 1]) / eval(lines[i]))
    cvr_csbdf.append(eval(lines[i + 3]) / eval(lines[i + 2]))
    cvr_sbucb.append(eval(lines[i + 5]) / eval(lines[i + 4]))

    ctcvr_dfm.append(eval(lines[i + 1]) / size)
    ctcvr_csbdf.append(eval(lines[i + 3]) / size)
    ctcvr_sbucb.append(eval(lines[i + 5]) / size)

    i += 7

cvr_dfm = np.array(cvr_dfm)
mean_dfm = np.mean(cvr_dfm)
var_dfm = np.var(cvr_dfm)
print('cvr_dfm', mean_dfm, var_dfm)

cvr_csbdf = np.array(cvr_csbdf)
mean_csbdf = np.mean(cvr_csbdf)
var_csbdf = np.var(cvr_csbdf)
print('cvr_csbdf', mean_csbdf, var_csbdf)

cvr_sbucb = np.array(cvr_sbucb)
mean_sbucb = np.mean(cvr_sbucb)
var_sbucb = np.var(cvr_sbucb)
print('cvr_sbucb', mean_sbucb, var_sbucb)

ctcvr_dfm = np.array(ctcvr_dfm)
mean_dfm = np.mean(ctcvr_dfm)
var_dfm = np.var(ctcvr_dfm)
print('ctcvr_dfm', mean_dfm, var_dfm)

ctcvr_csbdf = np.array(ctcvr_csbdf)
mean_csbdf = np.mean(ctcvr_csbdf)
var_csbdf = np.var(ctcvr_csbdf)
print('ctcvr_csbdf', mean_csbdf, var_csbdf)

ctcvr_sbucb = np.array(ctcvr_sbucb)
mean_sbucb = np.mean(ctcvr_sbucb)
var_sbucb = np.var(ctcvr_sbucb)
print('ctcvr_sbucb', mean_sbucb, var_sbucb)
