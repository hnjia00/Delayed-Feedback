import numpy as np
np.random.seed(1)
import math
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
import Artificial_data.config as config

cnt = 0

def train_dfm(X, y, timestamps, w_c, w_d, continuous=0, args={}):
    model = DFM(w_c=w_c, w_d=w_d, continuous=continuous,use_score_cache=False)

    model.fit(
            X=X,
            y=y,
            timestamps=timestamps, **args)

    return model

class DFM:
    """Delayed Feedback Model introducted by Chapelle(2014)"""

    def __init__(self, alpha=1, maxiter=50, eps=1e-9, w_c=0, w_d=0, continuous=0, use_score_cache=False):
        self.alpha = None
        self.timestamps = None
        self.alpha = alpha
        self.maxiter = maxiter
        self.eps = eps
        self.feature_num = config.d
        self.coef_ = np.zeros((2 * self.feature_num, 1))

        if continuous:
            self.coef_[:config.d] = np.reshape(w_c, (config.d, 1))
            self.coef_[config.d:] = np.reshape(w_d, (config.d, 1))

        self.use_score_cache = use_score_cache
        self.cached_scores = None

    def fit(self, X, y, timestamps):
        self.X = X
        self.y = y
        self.timestamps = timestamps

        initial_coef_ = np.zeros((2, self.feature_num))

        result = minimize(self.loss_func, self.coef_,
                          method='L-BFGS-B', jac=self.g, options={"maxiter": self.maxiter})
        self.coef_ = result['x']
        return self

    def predict(self, X):
        # TODO: there is a more efficient way because calc_scores calculates lambda which is not needed.
        scores = np.zeros(X.shape[0])
        w_c = np.reshape(self.coef_[:self.feature_num], (1, self.feature_num))
        for i in range(X.shape[0]):
            scores[i] = 1 / (1 + np.exp(-np.dot(w_c, X[i])))
        # scores = dfm.calc_scores_dfm(X, X_positions, self.num_features, self.coef_)
        # return scores[:, 0]
        return scores

    def dump(self, f):
        import pickle
        with open(f, "wb") as f:
            pickle.dump(self, f)

    def score(self, X, y):
        scores = np.zeros(X.shape[0])
        w_c = np.reshape(self.coef_[:self.feature_num], (1, self.feature_num))
        for i in range(X.shape[0]):
            scores[i] = 1 / (1 + np.exp(-np.dot(w_c, X[i])))
        # predicted_value = scores[:, 0]
        predicted_value = scores
        return log_loss(y, predicted_value), roc_auc_score(y, predicted_value)

    def g(self, coef_):
        scores = np.zeros(self.X.shape[0])
        lamda = np.zeros(self.X.shape[0])
        w_c = np.reshape(coef_[:self.feature_num], (1, self.feature_num))
        w_d = np.reshape(coef_[self.feature_num:], (1, self.feature_num))
        for i in range(self.X.shape[0]):
            scores[i] = 1 / (1 + np.exp(-np.dot(w_c, self.X[i])))
            lamda[i] = np.exp(np.dot(w_d, self.X[i]))

        coef_ = np.reshape(coef_, (2 * self.feature_num, 1))
        gradient = np.zeros((2 * self.feature_num, 1))
        for i in range(self.X.shape[0]):
            p = scores[i]
            l = lamda[i]
            coeff_c = p * (1 - p)
            coeff_d = l
            t = self.timestamps[i]
            if self.y[i]:
                coeff_c *= -1.0 / p
                coeff_d *= t - 1.0 / l
            else:
                coeff_c *= (1 - math.exp(-l * t)) / (1 - p + p * math.exp(-l * t))
                coeff_d *= (p * t * math.exp(-l * t)) / (1 - p + p * math.exp(-l * t))
            gradient[:self.feature_num] += coeff_c * np.reshape(self.X[i], (self.feature_num, 1))
            gradient[self.feature_num:] += coeff_d * np.reshape(self.X[i], (self.feature_num, 1))

        # gradient += self.alpha * coef_  # ？alpha * coef
        # gradient *= self.alpha
        return gradient

    def loss_func(self, coef_):
        # scores = 1 / (1 + np.dot(self.coef_[0], self.X))
        scores = np.zeros(self.X.shape[0])
        lamda = np.zeros(self.X.shape[0])
        w_c = np.reshape(coef_[:self.feature_num], (1, self.feature_num))
        w_d = np.reshape(coef_[self.feature_num:], (1, self.feature_num))
        for i in range(self.X.shape[0]):
            scores[i] = 1 / (1 + np.exp(-np.dot(w_c, self.X[i])))
            lamda[i] = np.exp(np.dot(w_d, self.X[i]))
        if self.use_score_cache:
            self.cached_scores = scores

        loss = self._empirical_loss(scores=scores, lamda=lamda)

        # 正则项
        # r = 0.5 * self.alpha * np.power(np.linalg.norm(coef_, 2), 2)
        r = 0.5 * self.alpha * (np.power(np.linalg.norm(w_c, 2), 2) + np.power(np.linalg.norm(w_d, 2), 2))

        return loss + r

    def _empirical_loss(self, scores, lamda):
        loss = 0
        for (label, p, l, t) in zip(self.y, np.clip(scores, self.eps, 1 - self.eps),
                                    np.clip(lamda, self.eps, 1 - self.eps), self.timestamps):
            loss -= int(label) * (math.log(p) + math.log(l) - l * t) + (1 - label) * math.log(
                1 - p + p * math.exp(-l * t))
            # if label:
            #     print(l*t)
        # print(loss)
        global cnt
        cnt += 1
        # print('ok', cnt)
        return loss

# 将输入的数据集D按动作的分属分割开
# {a,s,r,e,y}
def data_process(D_i):

    # 按动作排序
    D_i = D_i.tolist()
    D_i.sort(key=(lambda x:x[0]))
    D_i = np.array(D_i)

    S = []  # X
    e = []  # timestamps
    Y = []  # Y

    i = 0
    for a in config.A:
        S_i = []  # X
        e_i = []  # timestamps
        Y_i = []  # Y
        while i < len(D_i) and D_i[i][0] == a:
            S_i.append(D_i[i][1])
            e_i.append(D_i[i][3] if D_i[i][3] >= 0 else 0)
            Y_i.append(D_i[i][4])
            i += 1
        S.append(np.array(S_i))
        e.append(np.array(e_i))
        Y.append(np.array(Y_i))

    # (M, data_size/M): 维度（5, 2000）
    return S, Y, e
