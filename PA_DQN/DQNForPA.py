import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
import datetime
import sys
import os
import scipy.special
import heapq
dtype = np.float32
n_x = 4
n_y = 4
L = 2
C = 16
meanM = 2  # lamda: average user number in one BS
minM = 4  # maximum user number in one BS
maxM = 4  # maximum user number in one BS
maxUser = maxM
min_dis = 0.01  # km
max_dis = 1.  # km
min_p = 5.  # dBm
max_p = 38.  # dBm
p_n = -114.  # dBm

Ns = int(2e1)  # 训练时隙 40000
testNs = int(5e3)  # 测试时隙 5000
IN = 5  # 干扰者数量
IdN = 5  # 被干扰的邻居数量
state_dim = 7 + 6 * IN + 4 * IdN  # 状态维度
power_num = 10  # action_num
fd = 10
Ts = 20e-3
beta = 0.5
gama = 0.95  # 计算奖励时的权值
maxSinr = 30.  # 最大信噪比

c = n_x * n_y  # adjascent BS
I = c
K = maxM * c  # maximum adjascent users, including itself
state_num = 3 * C + 2  # 3*K - 1  3*C + 2
N = n_x * n_y  # BS number
M = N * maxM  # maximum users
O = M
W_ = np.ones((M), dtype=dtype)  # [M]
sigma2_ = 1e-3 * pow(10., p_n / 10.)  # 原文中的deta
maxP = 1e-3 * pow(10., max_p / 10.)
# 动作空间
power_set = np.hstack([np.zeros((1), dtype=dtype), 1e-3 * pow(10., np.linspace(min_p, max_p, power_num - 1) / 10.)])


# -----------------------------DQN参数-----------------------------------
# -----------------------------------------------------------------------
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class Config:
    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'DQN and DDQN'  # 算法名称
        self.env_name = 'MultiAgentDQNForPA'  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 10  # 随机种子，置0则不设置随机种子
        self.train_eps = 40000  # 训练的回合数
        self.test_eps = 5000  # 测试的回合数
        self.gamma = 0.5  # 强化学习中的折扣因子
        self.epsilon_start = 0.90  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 2000  # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # 学习率
        self.memory_capacity = 50000  # 经验回放的容量
        self.batch_size = 256  # mini-batch SGD中的批量大小
        self.target_update = 5  # 目标网络的更新频率
        self.hidden_dim1 = 200  # 网络隐藏层
        self.hidden_dim2 = 100
        self.hidden_dim3 = 40
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片


# -----------------------------------环境---------------------------------
# -----------------------------------------------------------------------
# 传入一个数组，找到其中前n大的数返回其下标
def getPriorNum(C_j, N):
    C_j = list(C_j)
    res1 = map(C_j.index, heapq.nlargest(N, C_j))
    res1 = list(res1)
    res = np.array(res1)
    return res


def Generate_H_set():
    '''
    Jakes model
    '''
    H_set = np.zeros([M, K, int(Ns)], dtype=dtype)
    pho = np.float32(scipy.special.k0(2 * np.pi * fd * Ts))
    H_set[:, :, 0] = np.kron(np.sqrt(0.5 * (np.random.randn(M, c) ** 2 + np.random.randn(M, c) ** 2)),
                             np.ones((1, maxM), dtype=np.int32))
    for i in range(1, int(Ns)):
        H_set[:, :, i] = H_set[:, :, i - 1] * pho + np.sqrt(
            (1. - pho ** 2) * 0.5 * (np.random.randn(M, K) ** 2 + np.random.randn(M, K) ** 2))
    path_loss = Generate_path_loss().T
    H2_set = np.square(H_set) * np.tile(np.expand_dims(path_loss, axis=2), [1, 1, int(Ns)])
    g_set = np.zeros([I, K, int(Ns)], dtype=dtype)
    for i in range(I):
        g_set[i] = H2_set[i*maxUser]
    return g_set


def Generate_environment():
    path_matrix = M * np.ones((n_y + 2 * L, n_x + 2 * L, maxM), dtype=np.int32)
    for i in range(L, n_y + L):
        for j in range(L, n_x + L):
            for l in range(maxM):
                path_matrix[i, j, l] = ((i - L) * n_x + (j - L)) * maxM + l
    p_array = np.zeros((M, K), dtype=np.int32)
    for n in range(N):
        i = n // n_x
        j = n % n_x
        Jx = np.zeros((0), dtype=np.int32)
        Jy = np.zeros((0), dtype=np.int32)
        for u in range(i - L, i + L + 1):
            v = 2 * L + 1 - np.abs(u - i)
            jx = j - (v - i%  2) // 2 + np.linspace(0, v - 1, num=v, dtype=np.int32) + L
            jy = np.ones((v), dtype=np.int32) * u + L
            Jx = np.hstack((Jx, jx))
            Jy = np.hstack((Jy, jy))
        for l in range(maxM):
            for k in range(c):
                for u in range(maxM):
                    p_array[n * maxM + l, k * maxM + u] = path_matrix[Jy[k], Jx[k], u]
    p_main = p_array[:, (c - 1) // 2 * maxM:(c + 1) // 2 * maxM]
    for n in range(N):
        for l in range(maxM):
            temp = p_main[n * maxM + l, l]
            p_main[n * maxM + l, l] = p_main[n * maxM + l, 0]
            p_main[n * maxM + l, 0] = temp
    p_inter = np.hstack([p_array[:, :(c - 1) // 2 * maxM], p_array[:, (c + 1) // 2 * maxM:]])
    p_array = np.hstack([p_main, p_inter])

    user = np.maximum(np.minimum(np.random.poisson(meanM, (N)), maxM), minM)
    user_list = np.zeros((N, maxM), dtype=np.int32)
    for i in range(N):
        user_list[i, :user[i]] = 1
    for k in range(N):
        for i in range(maxM):
            if user_list[k, i] == 0.:
                p_array = np.where(p_array == k * maxM + i, M, p_array)
    p_list = list()
    for i in range(M):
        p_list_temp = list()
        for j in range(K):
            p_list_temp.append([p_array[i, j]])
        p_list.append(p_list_temp)
    return p_array, p_list, user_list


def Generate_path_loss():
    slope = 0.  # 0.3
    p_tx = np.zeros((n_y, n_x))
    p_ty = np.zeros((n_y, n_x))
    p_rx = np.zeros((n_y, n_x, maxM))
    p_ry = np.zeros((n_y, n_x, maxM))
    dis_rx = np.random.uniform(min_dis, max_dis, size=(n_y, n_x, maxM))
    phi_rx = np.random.uniform(-np.pi, np.pi, size=(n_y, n_x, maxM))
    for i in range(n_y):
        for j in range(n_x):
            p_tx[i, j] = 2 * max_dis * j + (i % 2) * max_dis
            p_ty[i, j] = np.sqrt(3.) * max_dis * i
            for k in range(maxM):
                p_rx[i, j, k] = p_tx[i, j] + dis_rx[i, j, k] * np.cos(phi_rx[i, j, k])
                p_ry[i, j, k] = p_ty[i, j] + dis_rx[i, j, k] * np.sin(phi_rx[i, j, k])
    dis = 1e10 * np.ones((M, K), dtype=dtype)
    lognormal = np.zeros((M, K), dtype=dtype)
    for k in range(N):
        for l in range(maxM):
            for i in range(c):
                for j in range(maxM):
                    if p_array[k * maxM + l, i * maxM + j] < M:
                        bs = p_array[k * maxM + l, i * maxM + j] // maxM
                        dx2 = np.square((p_rx[k // n_x][k % n_x][l] - p_tx[bs // n_x][bs % n_x]))
                        dy2 = np.square((p_ry[k // n_x][k % n_x][l] - p_ty[bs // n_x][bs % n_x]))
                        distance = np.sqrt(dx2 + dy2)
                        dis[k * maxM + l, i * maxM + j] = distance
                        std = 8. + slope * (distance - min_dis)
                        lognormal[k * maxM + l, i * maxM + j] = np.random.lognormal(sigma=std)
    path_loss = lognormal * pow(10., -(120.9 + 37.6 * np.log10(dis)) / 10.)
    return path_loss

# 存储发射机i在t时刻的链路的信噪比  p_i:发散机i的功率   sinr[i][p_i][t]
def getSinr(g_set, P):
    sinr = np.zeros([I, 2], dtype=dtype)
    nr = 0.
    for t in range(2):
        for i in range(I):
                pow_i = power_set[P[t][i]]
                g_i_i = g_set[i][i*maxUser][t]
                gp = pow_i * g_i_i
                nr = 0.
                for j in range(I):
                    if j != i:
                        nr += g_set[j][i*maxUser][t] * power_set[P[t][j]]
                nr += sigma2_
                sinr[i][t] = np.minimum(gp/nr, maxSinr)
    return sinr


# 存储sinr的分母 发射机j在t时刻功率为pj对链路i 的噪声 Nr[i][pj][t]
def getNr(g_set):
    Nr = np.zeros([I, power_num, 2], dtype=dtype)
    for t in range(2):
        for i in range(I):
            sum_g = 0
            for pj in range(power_num):
                pow_j = power_set[pj]
                for j in range(I):
                    if j != i:
                        g_j_i = g_set[j][i*maxUser][t]
                        sum_g += g_j_i * pow_j
                Nr[i][pj][t] = sum_g + sigma2_
    return Nr


# 用户i在t时刻的下行频谱效率 C_set[i,p_i,t] p_i:为发散机i在t时刻的功率
def getC(sinr):
    C_set = np.zeros([I, Ns], dtype=dtype)
    for i in range(I):
        for t in range(2):
            if sinr[i][t] > 0.1:
                C_set[i][t] = np.log2(sinr[i][t] + 1)
    return C_set


# -------------------------------------环境交互--------------------------------------
# ----------------------------------------------------------------------------------
# 获取发射机i当前环境下的下一步状态 需要时刻t 发射机编号i 和 t-1时刻的所有功率 即上一时刻所有发射机的动作
# 返回下一步状态和奖励   P中存储的是0~t-1时刻所有发射机的动作的下标
# t1 为被干扰邻居的激活时间 这里设为t-1
def step(g_set, P, t, i, t1, C_, W):
    localIn, C_i, w_i = localInformation(g_set, P, t, i, C_)
    inIn = InterferingNeighbors(g_set, P, t, i, C_)
    idnIn = InterferedNeighbors(g_set, P, t, i, t1, C_)
    localIn = np.hstack([localIn, inIn])
    state = np.hstack([localIn, idnIn])
    # --------------------------计算奖励和效率
    if state[2] != 0:  # 如果C为0则下行链谱效率为0
        rate = state[1] * (1/state[2])  # 加权下行链谱效率
        # ------------------计算奖励
        # ---------得到邻居的集合
        # nr_k_t_1 = np.zeros((O), dtype=dtype)  # 存放所有t-1时刻 被干扰邻居的信噪比
        # for k in range(O):
        #     j = k // maxUser
        #     if j != i:
        #         gp = g_set[i][k][t - 1] * power_set[P[t - 1][i]]
        #         nr_t_1 = 0.
        #         for j_ in range(I):
        #             if j_ != i:
        #                 nr_t_1 += g_set[j_][k][t - 1] * power_set[P[t - 1][j_]]
        #         nr_t_1 += sigma2_
        #         nr_k_t_1[k] = np.minimum(gp / nr_t_1, maxSinr)
        # res = getPriorNum(nr_k_t_1, IdN)
        # pi_i_k = 0.
        # for index in range(IdN):
        #     k = res[index]
        #     gp = g_set[k//maxUser][k][t-1] * power_set[P[t-1][k//maxUser]]
        #     nr = 0.
        #     # ----计算k的信噪比
        #     for j in range(I):
        #         if j != k//maxUser:
        #             nr += g_set[j][k][t-1] * power_set[P[t-1][j]]
        #     nr += sigma2_
        #     C_k = np.log2(1+gp/nr)
        #     # ----计算k中没有i的干扰时的信噪比
        #     nr -= g_set[i][k][t-1] * power_set[P[t-1][i]]
        #     C_k_i = np.log2(1+gp/nr)
        #     # ----得到k在t-1时刻的权值
        #     w_k = W[t-1][k//maxUser]
        #     pi_i_k += w_k * (C_k_i - C_k)
        reward = (state[1] / state[2]) - C_set[i][t-1] * W[t-1][i]  # C_set[i][t-1] * W[t-1][i]  pi_i_k * gama
    else:
        reward = 0.
        rate = 0.
    return state, reward, C_i, rate, w_i


# 获取链路i的本地信息 共7个
def localInformation(g_set, P, t, i, C_):
    localIn = []
    p_i_t_1 = power_set[P[t-1][i]]
    #----------添加t-1时刻的功率 如果为0则不做功
    localIn.append(p_i_t_1)
    # ----------添加t-1时刻发射机i的下行链谱效率 : 在state中的第2个元素
    nr_i_t_1 = 0.
    g_i_i = g_set[i][i*maxUser][t-1]
    gp = g_i_i * p_i_t_1
    for j in range(I):
        if j != i:
            nr_i_t_1 += g_set[j][i*maxUser][t-1] * power_set[P[t-1][j]]
    nr_i_t_1 += sigma2_
    sinr_i_t_1 = np.minimum(gp/nr_i_t_1, maxSinr)
    C_i_t_1 = maxUser*np.log2(1+sinr_i_t_1)
    localIn.append(C_i_t_1)
    C_set[i][t] = C_i_t_1
    #----------添加t-1时刻发射机i的权值 : 在state中的第3个元素
    C_i = beta * C_i_t_1 + (1-beta)*C_[t-1][i]  # 计算权值的C_
    W_i = 0.
    if C_i != 0:
        W_i = 1/C_i
    localIn.append(C_i)
    #----------添加t时刻g_i_i
    localIn.append(g_set[i][i*maxUser][t])
    # ----------添加t-1时刻g_i_i
    localIn.append(g_set[i][i * maxUser][t-1])
    # 计算t-1时刻和t-2时刻的噪声
    nr_i_t_2 = 0.
    for j in range(I):
        if j != i:
            nr_i_t_2 += g_set[j][i * maxUser][t - 2] * power_set[P[t - 2][j]]
    nr_i_t_2 += sigma2_
    localIn.append(nr_i_t_1)
    localIn.append(nr_i_t_2)
    return localIn, C_i, W_i


# 获取链路i的干扰者信息 共 IN*6 个
def InterferingNeighbors(g_set, P, t, i, C_):
    inIn = []
    gp_1 = np.zeros((I), dtype=dtype) # 存放所有t-1时刻干扰者的gp
    gp_2 = np.zeros((I), dtype=dtype) # 存放所有t-2时刻干扰者的gp
    for j in range(I):
        if j != i:
            gp_1[j] = g_set[j][i*maxUser][t-1] * power_set[P[t-1][j]]
            gp_2[j] = g_set[j][i*maxUser][t-2] * power_set[P[t-2][j]]
    res1 = getPriorNum(gp_1, IN)
    res2 = getPriorNum(gp_2, IN)
    # -------------添加所有干扰者t-1时刻的信息
    for index in range(IN):
        j1 = res1[index]
        # ----------添加所有干扰者j,t-1时刻的g*p
        inIn.append(g_set[j1][i*maxUser][t]*power_set[P[t-1][j1]])
        # ----------添加所有干扰者j,t-1时刻的下行链谱效率
        nr_j1_t_1 = 0.
        g_j1_j1 = g_set[j1][j1 * maxUser][t - 1]
        gp = g_j1_j1 * power_set[P[t-1][j1]]
        for j in range(I):
            if j != j1:
                nr_j1_t_1 += g_set[j][j1 * maxUser][t - 1] * power_set[P[t - 1][j]]
        nr_j1_t_1 += sigma2_
        sinr_j1_t_1 = np.minimum(gp / nr_j1_t_1, maxSinr)
        C_j1_t_1 = maxUser*np.log2(1 + sinr_j1_t_1)
        inIn.append(C_j1_t_1)
        # ----------添加所有干扰者j,t-1时刻的权值
        C_j1 = beta * C_j1_t_1 + (1 - beta) * C_[t-1][j1]  # 计算权值的C_
        # W_j1_t_1 = 1 / C_j1
        inIn.append(C_j1)
    # -------------添加所有干扰者t-2时刻的信息
    for index in range(IN):
        j2 = res2[index]
        # ----------添加所有干扰者j,t-1时刻的g*p
        inIn.append(g_set[j2][i * maxUser][t-1] * power_set[P[t - 2][j2]])
        # ----------添加所有干扰者j,t-1时刻的下行链谱效率
        nr_j2_t_2 = 0.
        g_j2_j1 = g_set[j2][j2 * maxUser][t - 2]
        gp = g_j2_j1 * power_set[P[t - 2][j2]]
        for j in range(I):
            if j != j2:
                nr_j2_t_2 += g_set[j][j2 * maxUser][t - 2] * power_set[P[t - 2][j]]
        nr_j2_t_2 += sigma2_
        sinr_j2_t_2 = np.minimum(gp / nr_j2_t_2, maxSinr)
        C_j2_t_2 = maxUser*np.log2(1 + sinr_j2_t_2)
        inIn.append(C_j2_t_2)
        # ----------添加所有干扰者j,t-2时刻的权值
        C_j2 = beta * C_j2_t_2 + (1 - beta) * C_[t-2][j2]  # 计算权值的C_
        # W_j2_t_2 = 1 / C_j2
        inIn.append(C_j2)
    return inIn


# 获取链路i的被干扰邻居的信息 共 IdN*4 个
def InterferedNeighbors(g_set, P, t, i, t1, C_):
    idnIn = []
    nr_k_t_1 = np.zeros((O), dtype=dtype)  # 存放所有t-1时刻 被干扰邻居的信噪比
    for k in range(O):
        j = k//maxUser
        if j != i:
            gp = g_set[i][k][t-1] * power_set[P[t-1][i]]
            nr_t_1 = 0.
            for j_ in range(I):
                if j_ != i:
                    nr_t_1 += g_set[j_][k][t - 1] * power_set[P[t - 1][j_]]
            nr_t_1 += sigma2_
            nr_k_t_1[k] = np.minimum(gp/nr_t_1, maxSinr)
    res = getPriorNum(nr_k_t_1, IdN)
    for index in range(IdN):
        k = res[index]
        trs = k//maxUser  # 邻居所在的区域
        # ----------------添加被干扰邻居的信息
        # --------添加g_k_k
        g_k_k = g_set[trs][k][t - 1]
        idnIn.append(g_k_k)
        # --------添加k在 t-1时刻的下行链谱效率
        gp = g_k_k * power_set[P[t - 1][trs]]
        nr_t_1 = 0.
        for j in range(I):
            if j != trs:
                nr_t_1 += g_set[j][k][t - 1] * power_set[P[t - 1][j]]
        nr_t_1 += sigma2_
        sinr_t_1 = np.minimum(gp/nr_t_1, maxSinr)
        C_K_t_1 = np.log2(1+sinr_t_1)
        idnIn.append(C_K_t_1)
        # --------添加k在 t-1时刻的权值
        C_k = beta * C_K_t_1 + (1 - beta) * C_[t-1][trs]  # 计算权值的C_
        # W_k = 1 / C_k
        idnIn.append(C_k)
        # --------添加k在 t1时刻的信噪比
        if (t1 >= 0) and (t1 < t) and (power_set[P[t1][i]] != 0):
            gp = g_set[i][k][t1] * power_set[P[t1][i]]
        else: gp = 0
        sinr_t_1 = np.minimum(gp / nr_t_1, maxSinr)
        idnIn.append(sinr_t_1)
    return idnIn


# 初始化 t=1 和 t=2时的状态state 以及 这两个时刻动作，t从2开始
def initEnv(C_set):
    states = []
    rewards = []
    C_ = np.zeros((Ns, I), dtype=dtype)  # C_[0][i]为：t时刻权值的倒数
    sumRate = np.zeros((Ns, I), dtype=dtype)  # sumRate[t][i]为：t时刻 发射机i的加权下行链谱效率
    W = np.zeros((Ns, I), dtype=dtype)  # W[t][i]为: t时刻 发射机i的权值
    for j in range(I):
        C_[0][j] = C_set[j][0]
        C_[1][j] = beta * C_set[j][1] + (1-beta) * C_[0][j]
        if C_[0][j] != 0:
            sumRate[0][j] = 1 / C_[0][j] * C_set[j][0]
            W[0][j] = 1/C_[0][j]
        if C_[1][j] != 0:
            sumRate[1][j] = 1 / C_[1][j] * C_set[j][1]
            W[1][j] = 1/C_[1][j]
    for i in range(I):
        state, reward, C_i, rate, w_i = step(g_set, P, 2, i, 1, C_, W)
        sumRate[2][i] = rate
        C_[2][i] = C_i
        W[2][i] = w_i
        states.append(state)
        rewards.append(reward)
    return states, rewards, C_, sumRate, W


# -----------------------------------DQN---------------------------------
# -----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=200, hidden_dim2=100, hidden_dim3=40):
        """ 初始化q网络，为全连接网络
            state_dim: 输入的特征数即环境的状态维度
            action_dim: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)  # 输入层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)  # 隐藏层
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim3)  # 隐藏层
        self.fc5 = nn.Linear(hidden_dim3, action_dim)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # 刚开始例表内没东西，不能通过下标索引到空间，添加None后可以通过position进行索引
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采出小批量转移
        state, action, reward, next_state = zip(*batch)  # 解压成状态，动作等
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim  # 总的动作个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim, hidden_dim1=cfg.hidden_dim1, hidden_dim2=cfg.hidden_dim2,
                              hidden_dim3=cfg.hidden_dim3).to(self.device)
        self.target_net = MLP(state_dim, action_dim, hidden_dim1=cfg.hidden_dim1, hidden_dim2=cfg.hidden_dim2,
                              hidden_dim3=cfg.hidden_dim3).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)  # 优化器
        self.memory = ReplayBuffer(cfg.memory_capacity)  # 经验回放

    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = np.array(state)
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
                # print(q_values)
                action = q_values.max(1)[1].item()  # 选择Q值最大的动作
                # print(action)
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample(
            self.batch_size)
        # 转为张量
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64).unsqueeze(1)
        # print(action_batch)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # 计算当前状态(s_t,a)对应的Q(s_t, a)
        # print(q_values)
        # #--------- DDQN ---------  收敛数度快了一倍
        # q_values_next_state = self.policy_net(next_state_batch)
        # next_action = q_values_next_state.max(1)[1].unsqueeze(1)
        # next_q_values = self.target_net(next_state_batch).gather(dim=1, index=next_action)
        # next_q_values = next_q_values.reshape(1, self.batch_size).squeeze(0)
        # #--------- DQN -----------
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()  # 计算下一时刻的状态(s_t_,a)对应的Q值
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


def agent_config(cfg):
    action_dim = power_num  # 动作维度
    agent = DQN(state_dim, action_dim, cfg)  # 创建智能体
    return agent


# 执行t=1时的动作返回其t=2时所有发射机i的状态states[i] 得到t=2时的所有奖励rewards[i]
# 以及返回 初始时执行的动作P[0][]以及P[1][]
# sumRate[t][i]为：t时刻 发射机i的加权下行链谱效率
# W[t][i]为: t时刻 发射机i的权值
# C_[t][i]为：t时刻 发射机i权值的倒数
def train(cfg, agent):
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    SumRate_ep = np.zeros([cfg.train_eps], dtype=dtype)
    ma_rewards_ep = []
    states_ = []
    for i_ep in range(cfg.train_eps):
        ep_reward = []  # 记录每一时刻发射机的奖励
        ma_rewards = []  # 记录所有时隙的滑动平均奖励
        if i_ep == 0:
            states, rewards, C_, sumRate, W = initEnv(C_set)  # 重置环境，返回初始状态
            ep_reward.append(sum(rewards))  # 将执行动作P[1][i]的奖励存储起来
        else:
            for t in range(2):
                for i in range(I):
                    action = agent.choose_action(states_[i])
                    P[0][i] = action
            sinr = getSinr(g_set, P)
            c_set = getC(sinr)
            states, rewards, C_, sumRate, W = initEnv(c_set)
        sumTempRate = (sum(sumRate[0]) + sum(sumRate[1]))
        sumTempReward = sum(rewards)
        for t in range(3, Ns):
            for i in range(I):
                # 为每一个发射机选择动作
                action = agent.choose_action(states[i])
                P[t-1][i] = action
            # 执行动作
            rewards = []
            for i in range(I):
                next_state, reward, C_i, rate, w_i = step(g_set, P, t, i, t-1, C_, W)  # 更新环境，返回transition
                agent.memory.push(states[i], P[t-1][i], reward, next_state)  # 保存transition
                sumRate[t][i] = rate
                C_[t][i] = C_i
                W[t][i] = w_i
                states[i] = next_state
                rewards.append(reward)
            ep_reward.append(sum(rewards))
            agent.update()  # 更新智能体
            if (t - 2) % cfg.target_update == 0:  # 智能体目标网络更新
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * sum(rewards))
            else:
                ma_rewards.append(sum(rewards))
            sumTempRate += sum(sumRate[t])
            sumTempReward += sum(rewards)
        states_ = states.copy()
        print('回合：{}/{}, 平均和率为：{}, 奖励：{}'.format(i_ep, cfg.train_eps, sumTempRate/Ns, sumTempReward/Ns))
        print('完成第%d次训练！' % (i_ep+1))
        if i_ep > 0:
            SumRate_ep[i_ep] = (0.1 * sumTempRate / Ns) + 0.9 * SumRate_ep[i_ep-1]
        else:
            SumRate_ep[i_ep] = sumTempRate / Ns
        ma_rewards_ep.append(ma_rewards)
    return SumRate_ep, ma_rewards_ep


if __name__ == "__main__":
    p_array, p_list, user_list = Generate_environment()
    path_loss = Generate_path_loss()
    g_set = Generate_H_set()
    print(power_set)
    power_set = np.linspace(0., 6.30957344, power_num)
    print(power_set)
    cfg = Config()
    agent = agent_config(cfg)
    P = np.zeros((Ns, I), dtype=int)
    P_t = np.random.randint(0, power_num, size=(1, I)).reshape(I, )
    P_t_1 = np.random.randint(0, power_num, size=(1, I)).reshape(I, )
    P[0] = P_t  # 0时刻所有发射机的动作
    P[1] = P_t_1  # 1时刻所有发射机的动作
    Nr = getNr(g_set)
    sinr = getSinr(g_set, P)
    C_set = getC(sinr)
    states, rewards, C_, sumRate, W = initEnv(C_set)
    SumRate_ep, ma_rewards_ep = train(cfg, agent)
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('时隙')
    plt.plot(SumRate_ep, label='平均和率')
    plt.show()
