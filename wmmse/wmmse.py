from numpy import *
import time
import scipy.special

# matlab比python快了4倍
def findClusterClosures(clusterLocations, radius):
    m = shape(clusterLocations)[0]
    closures = mat(zeros((m, m)))
    for k in range(m):
        for k1 in range(m):
            if abs(clusterLocations[k] - clusterLocations[k1]) <= radius:
                closures[k, k1] = k1 + 1
    return closures


def brownian(K, Q, I, locations, outerRadius):
    BSs = mat(zeros((K * Q, 1), dtype=complex))
    UEs = mat(zeros((K * I, 1), dtype=complex))
    for k in range(K):
        for q in range(Q):
            BSs[k * Q + q] = locations[k]
        for i in range(I):
            while True:
                x = (round(random.random(), 4) - 0.5) * 2 * outerRadius
                y = (round(random.random(), 4) - 0.5) * 2 * outerRadius
                mx = abs(x)
                my = abs(y)
                valid = True
                if my > outerRadius * sin(pi / 3) or (sqrt(3) * mx + my > outerRadius * sqrt(3)):
                    valid = False
                if valid:
                    break
            UEs[k * I + i] = x + y * 1j + locations[k]
    return BSs, UEs


def model2(N, M, d):
    L = pow(10, round(random.normal(0, 1), 4) * 8 / 10)
    sigma = sqrt(pow(200 / d, 3) * L)
    h = multiply((random.normal(0, 1, (N, M)) + multiply(random.normal(0, 1, (N, M)), 1j)), sigma)
    return h


def generateMIMOChannel(K, Q, M, bsLocations, I, N, ueLocations, model):
    H = mat(zeros((K * I * N, K * Q * M), dtype=complex))
    if model == 1:
        for i in range(shape(H)[0]):
            for j in range(shape(H)[1]):
                H[i, j] = (round(random.normal(0, 1), 4) + round(random.normal(0, 1), 4) * 1j) / sqrt(2)
    if model == 2:
        for k1 in range(K):
            for q in range(Q):
                colOffset = k1 * Q * M + q * M  # 列坐标，发射机的坐标
                for k2 in range(K):
                    for i in range(I):
                        # 计算基站k1中的发射机q到区域k2中的用户i的距离
                        d = abs(bsLocations[k1 * Q + q] - ueLocations[k2 * I + i])
                        rowOffset = k2 * I * N + i * N
                        H[rowOffset:rowOffset + N, colOffset:colOffset + M] = model2(N, M, d)
    return H


def waterfill(x, P):
    L = shape(x)[0]
    y = sort(x, axis=0)
    a, b = shape(y)
    if a > b:
        y = y.transpose()
        x = x.transpose()
    delta = mat(zeros((1, L), dtype=complex))
    delta[0, 1:] = multiply(cumsum(y[0, 1:] - y[0, 0:L - 1]), mat(range(1, L)))
    IndexArr = where(delta <= P)[1]
    I = IndexArr[-1]  # 找到最后一个<=P的数在delta的下标
    if I == 0: I = 1
    level = y[0, I] + (P - delta[0, I]) / I
    x = array(x)
    p = (abs(level - x) + level - x) / 2
    p = mat(p)
    return p


def generateRandomTxVector(K, Q, M, I, N, P, H, closures, model):
    V = mat(zeros((K * Q * M, K * I), dtype=complex))  # 波束矩阵，存放每台发射机到每个用户的波束信息
    A = mat(zeros((K * Q, K * I), dtype=complex))  # 功率矩阵，存放每台发射机发给对应用户消耗的功率信息
    for k1 in range(K):
        closure = closures[k1, :]  # 区域k1中的发射机可以给哪些区域发信息非0则可以发
        numUEs = len(closure.nonzero()[0]) * I  # 这些区域的接收机数量
        closure = array(closure)[0]
        for q in range(Q):
            p = mat(zeros((K * I, 1)))  # 存放发射机q到每个接收机的功率
            effNoise = multiply(mat(ones((K * I, 1), dtype=complex)), 1e100)
            for k in closure:
                k = int(k)
                if k == 0: continue
                for i in range(I):
                    rowOffset = (k - 1) * I * N + i * N
                    colOffset = k1 * Q * M + q * M
                    h = H[rowOffset:rowOffset + N, colOffset: colOffset + M]
                    effNoise[(k - 1) * I + i] = 1 / trace(h @ h.H)
            p = waterfill(effNoise, P).transpose()
            for k in closure:
                k = int(k)
                if k == 0: continue
                for i in range(I):
                    v = mat(random.normal(0, 1, (M, 1))) + multiply(random.normal(0, 1, (M, 1)), 1j)
                    if model == 1:
                        Power = P / numUEs  # 功率预算
                    if model == 2:
                        Power = p[(k - 1) * I + i]  # 功率按照充水算法分配
                    v = multiply(v / linalg.norm(v, ord=2, axis=0, keepdims=True), sqrt(Power))
                    rowOffset = k1 * Q * M + q * M
                    colOffset = (k - 1) * I + i
                    # 更新区域k1中第q台发射机到区域k第i个用户的 V
                    V[rowOffset:rowOffset + M, colOffset] = v
                    A[k1 * Q + q, (k - 1) * I + i] = Power
    return V, A


def updateWMMSEVariables(K, Q, M, I, N, H, V):
    U = mat(zeros((K * I * N, 1), dtype=complex))
    W = mat(zeros((K * I, 1), dtype=complex))
    R = mat(zeros((K * I, 1)))
    for k in range(K):
        for i in range(I):
            C = mat(zeros((N, N), dtype=complex))
            for k1 in range(K):
                rowOffset = k * I * N + i * N
                colOffset = k1 * Q * M
                h = H[rowOffset:rowOffset + N, colOffset:colOffset + Q * M]  # k区域的用户i到k1区域所有发射机的信道信息
                rowOffset = k1 * Q * M
                v = V[rowOffset:rowOffset + Q * M, :]  # k1区域的发射机到所有用户的波束信息
                hv = h @ v
                C = C + hv @ hv.H
            C = C + mat(eye(N, N))
            rowOffset = k * I * N + i * N
            colOffset = k * Q * M
            h = H[rowOffset:rowOffset + N, colOffset:colOffset + Q * M]  # k区域的用户i到k区域所有发射机的信道信息
            rowOffset = k * Q * M
            colOffset = k * I + i
            v = V[rowOffset:rowOffset + Q * M, colOffset]  # k1区域的发射机到k区域用户i的波束信息
            localHv = h @ v
            offset = k * I * N + i * N
            U[offset:offset + N, 0] = linalg.pinv(C) @ localHv
            W[k * I + i, 0] = 1 / (1 - real(localHv.H @ U[offset:offset + N, 0]) + 1e-5)
            L = C - localHv * localHv.H
            R[k * I + i, :] = log2(real(linalg.det(mat(eye(N, N)) + localHv * localHv.H * linalg.pinv(L))) + 1e-5)
    return U, W, R


def updateMmseMMatrix(K, Q, M, I, N, H, U, W):
    mmse = mat(zeros((K * Q * M, Q * M), dtype=complex))
    for k in range(K):
        m = mat(zeros((Q * M, Q * M)), dtype=complex)
        for k1 in range(K):
            for i in range(I):
                rowOffset = k1 * I * N + i * N
                colOffset = k * Q * M
                h = H[rowOffset: rowOffset + N, colOffset: colOffset + Q * M]
                offset = k1 * I * N + i * N
                u = U[offset: offset + N, 0]
                hu = h.H * u
                m = m + multiply(W[k1 * I + i, 0], hu * hu.H)
        offset = k * Q * M
        mmse[offset: offset + Q * M, :] = m
    return mmse


def arrayToMatrix(D):
    n = len(D)
    M = mat(zeros((n, n), dtype=complex))
    for i in range(n):
        M[i, i] = D[i]
    return M


# 求解miu所用的函数
def mmseBisectionTarget(phi, Lambda, multiplier):
    p = 0
    for i in range(shape(phi)[0]):
        p = p + real(phi[i, i]) / power(real(Lambda[i, i] + multiplier), 2)
    return p


def iterateWMMSE(K, Q, M, I, N, mmse, P, H, W, U):
    V = mat(zeros((K * Q * M, K * I), dtype=complex))
    for k in range(K):
        offset = k * Q * M
        MMatrix = mmse[offset: offset + Q * M, :]
        Power = 0
        tmp = mat(zeros((Q * M, Q * M), dtype=complex))
        for i in range(I):
            rowOffset = k * I * N + i * N
            colOffset = k * Q * M
            h = H[rowOffset: rowOffset + N, colOffset: colOffset + Q * M]
            offset = k * I * N + i * N
            u = U[offset: offset + N, :]
            w = W[k * I + i, 0]
            v = linalg.pinv(MMatrix) * multiply(h.H * u, w)
            rowOffset = k * Q * M
            colOffset = k * I + i
            V[rowOffset:rowOffset + Q * M, colOffset] = v
            Power = Power + power(linalg.norm(v, ord=2, axis=0, keepdims=True), 2)
            tmp = tmp + multiply(h.H * (u * u.H) * h, w * w)
        # 第一种情况miu=0时在条件区域内取最小值 继续更新
        if Power <= P * Q:
            continue
        # 第二种情况miu>0 在条件处取最小值
        Lambda, D = linalg.eig(MMatrix)  # 将H'UWU'H特征分解返回特征值的对角矩阵 Lambda 和特征向量矩阵 D
        Lambda = arrayToMatrix(Lambda)
        phi = D.H * tmp * D
        # 求解miu
        miuLow = 0
        miuHigh = 1
        while mmseBisectionTarget(phi, Lambda, miuHigh) > P * Q:  # 逼近miu
            miuHigh = miuHigh * 2
        multiplier = (miuHigh + miuLow) / 2
        targetValue = mmseBisectionTarget(phi, Lambda, multiplier)
        while abs(targetValue - P * Q) / (P * Q) >= 1e-14:
            if targetValue > P * Q:
                miuLow = multiplier
            elif targetValue < P * Q:
                miuHigh = multiplier
            else:
                break
            multiplier = (miuHigh + miuLow) / 2
            targetValue = mmseBisectionTarget(phi, Lambda, multiplier)
        # 开始更新V
        offset = k * Q * M
        MMatrix = mmse[offset: offset + Q * M, :]
        for i in range(I):
            rowOffset = k * I * N + i * N
            colOffset = k * Q * M
            h = H[rowOffset:rowOffset + N, colOffset:colOffset + Q * M]
            offset = k * I * N + i * N
            u = U[offset:offset + N, :]
            w = W[k * I + i, :]
            v = multiply(linalg.pinv(multiply(multiplier, mat(eye(Q * M, Q * M))) + MMatrix) * h.H * u,
                         w)  # 计算第k个区域发射机对用户i的最优波束
            rowOffset = k * Q * M
            colOffset = k * I + i
            V[rowOffset:rowOffset + Q * M, colOffset] = v
    return V


if __name__ == "__main__":
    K = 4  # 区域数
    M = 1  # 发射机天线数
    N = 1  # 接收机天线数
    Q = 1  # 每个基站发射机个数
    I = 4  # 每个区域用户个数
    fd = 10
    Ts = 20e-3
    SNRdB = 25
    SNR = power(10, SNRdB / 10)
    P = SNR / Q  # 每台发射机的功率预算
    r = 1000  # 基站能给接收机发送信息的最远距离
    clusterLocations = mat([0 + 0j, 0 + r * 1j, r * cos(pi / 6) + r * sin(pi / 6) * 1j,
                            - r * cos(pi / 6) + r * sin(pi / 6) * 1j]).transpose()  # 记录区域与区域之间的距离
    closures = findClusterClosures(clusterLocations, r)
    # print(closures)
    bsLocations, ueLocations = brownian(K, Q, I, clusterLocations, r / sqrt(3))  # 获取基站与用户的位置信息
    # H = generateMIMOChannel(K, Q, M, bsLocations, I, N, ueLocations, 2)
    # print(H)
    # 开始仿真
    numCases = 10  # 进行10次模拟
    totalSumRate = 0  # 计算总和率
    totalNumIterations = 0  # 计算总迭代次数
    maxIterations = 50  # 控制最大迭代次数
    epsilon = 1e-3  # 误差：迭代出口
    pho = float32(scipy.special.k0(2 * pi * fd * Ts))
    startTime = time.time()
    for i in range(numCases):  # 计算10个案例求平均值
        # 随机初始化
        numIterations = 0
        prev = 0
        [bsLocations, ueLocations] = brownian(K, Q, I, clusterLocations, r / sqrt(3))
        # 初始化信道信息
        H = generateMIMOChannel(K, Q, M, bsLocations, I, N, ueLocations, 2)
        V, A = generateRandomTxVector(K, Q, M, I, N, P, H, closures, 2)
        # print(A.shape)
        U, W, R = updateWMMSEVariables(K, Q, M, I, N, H, V)
        # mmse = updateMmseMMatrix(K, Q, M, I, N, H, U, W)
        # print(mmse)
        sumR = sum(R)
        # print(sumR)
        # 开始迭代
        while abs(prev - sumR) > epsilon:  # abs(prev - sumR) > epsilon
            prev = sumR
            numIterations = numIterations + 1
            if numIterations > maxIterations:
                numIterations = numIterations - 1
                break
            mmse = updateMmseMMatrix(K, Q, M, I, N, H, U, W)
            # H = generateMIMOChannel(K, Q, M, bsLocations, I, N, ueLocations, 2)
            V = iterateWMMSE(K, Q, M, I, N, mmse, P, H, W, U)
            # H = generateMIMOChannel(K, Q, M, bsLocations, I, N, ueLocations, 2)
            [U, W, R] = updateWMMSEVariables(K, Q, M, I, N, H, V)
            sumR = sum(R)
            H = H + mat(sqrt(
                (1. - pho ** 2) * 0.5 * (
                            random.randn(K * I * N, K * Q * M) ** 2 + random.randn(K * I * N, K * Q * M) ** 2)))
            print('在案例 %d 中：第 %d 次迭代中，平均和率R = %f' % (i, numIterations, sumR/4))
        print('在案例 %d 中：共迭代了 %d 次 ,平均和率R = %f' % (i, numIterations, sumR/4))
        totalSumRate = totalSumRate + sumR
        totalNumIterations = totalNumIterations + numIterations
    endTime = time.time()
    print('总平均和率为 : %f' % (totalSumRate / numCases / 4))
    print('平均迭代次数为 : %f' % (totalNumIterations / numCases))
    print('执行时间为%f 秒' % (endTime - startTime))
