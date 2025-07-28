from numpy import diag as diags
from numpy.linalg import solve
import numpy as np
import scipy
from scipy.sparse import csc_matrix
import time

# 禁用JAX相关配置，使用NumPy的64位精度
density_filtering = False
sensitivity_filtering = False


def compute_filter_kd_tree(fe):
    """使用SciPy稀疏矩阵替代JAX的BCOO稀疏矩阵"""
    cell_centroids = np.mean(np.take(fe.points, fe.cells, axis=0), axis=1)
    flex_num_cells = len(fe.flex_inds)
    flex_cell_centroids = np.take(cell_centroids, fe.flex_inds, axis=0)

    V = np.sum(fe.JxW)
    avg_elem_V = V / fe.num_cells

    avg_elem_size = avg_elem_V ** (1. / fe.dim)
    rmin = 1.5 * avg_elem_size

    kd_tree = scipy.spatial.KDTree(flex_cell_centroids)
    I = []
    J = []
    V_vals = []
    for i in range(flex_num_cells):
        num_nbs = 20
        dd, ii = kd_tree.query(flex_cell_centroids[i], num_nbs)
        vals = np.where(rmin - dd > 0., rmin - dd, 0.)
        I += [i] * num_nbs
        J += ii.tolist()
        V_vals += vals.tolist()
    # 使用SciPy的CSC稀疏矩阵替代JAX的BCOO
    H_sp = csc_matrix((V_vals, (I, J)), shape=(flex_num_cells, flex_num_cells), dtype=np.float64)
    # 计算行和（替代JAX的sum(1)）
    Hs = H_sp.sum(axis=1).A.flatten()  # A属性转为 dense array
    return H_sp, Hs


def applySensitivityFilter(ft, rho, dJ, dvc):
    """使用SciPy稀疏矩阵乘法替代JAX的@操作"""
    # 稀疏矩阵与稠密矩阵乘法：H.dot(...)
    dJ = ft['H'].dot(rho * dJ / np.maximum(1e-3, rho) / ft['Hs'][:, None])
    # 处理约束梯度的三维情况
    rho_3d = rho[None, :, :]
    dvc_filtered = ft['H'].dot((rho_3d * dvc / np.maximum(1e-3, rho_3d) / ft['Hs'][None, :, None]).squeeze())
    return dJ, dvc_filtered.reshape(dvc.shape)


def applyDensityFilter(ft, rho):
    """使用SciPy稀疏矩阵乘法替代JAX的@操作"""
    return ft['H'].dot(rho) / ft['Hs'][:, None]


# %% Optimizer
class MMA:
    # 基于MMA算法的优化器（移除JAX依赖）
    def __init__(self):
        self.epoch = 0;

    def resetMMACounter(self):
        self.epoch = 0;

    def registerMMAIter(self, xval, xold1, xold2):
        self.epoch += 1;
        self.xval = xval;
        self.xold1 = xold1;
        self.xold2 = xold2;

    def setNumConstraints(self, numConstraints):
        self.numConstraints = numConstraints;

    def setNumDesignVariables(self, numDesVar):
        self.numDesignVariables = numDesVar;

    def setMinandMaxBoundsForDesignVariables(self, xmin, xmax):
        self.xmin = xmin;
        self.xmax = xmax;

    def setObjectiveWithGradient(self, obj, objGrad):
        self.objective = obj;
        self.objectiveGradient = objGrad;

    def setConstraintWithGradient(self, cons, consGrad):
        self.constraint = cons;
        self.consGrad = consGrad;

    def setScalingParams(self, zconst, zscale, ylinscale, yquadscale):
        self.zconst = zconst;
        self.zscale = zscale;
        self.ylinscale = ylinscale;
        self.yquadscale = yquadscale;

    def setMoveLimit(self, movelim):
        self.moveLimit = movelim;

    def setLowerAndUpperAsymptotes(self, low, upp):
        self.lowAsymp = low;
        self.upAsymp = upp;

    def getOptimalValues(self):
        return self.xmma, self.ymma, self.zmma;

    def getLagrangeMultipliers(self):
        return self.lam, self.xsi, self.eta, self.mu, self.zet;

    def getSlackValue(self):
        return self.slack;

    def getAsymptoteValues(self):
        return self.lowAsymp, self.upAsymp;

    # MMA子问题求解
    def mmasub(self, xval):
        m = self.numConstraints;
        n = self.numDesignVariables;
        iter = self.epoch;
        xmin, xmax = self.xmin, self.xmax;
        xold1, xold2 = self.xold1, self.xold2;
        f0val, df0dx = self.objective, self.objectiveGradient;
        fval, dfdx = self.constraint, self.consGrad;
        low, upp = self.lowAsymp, self.upAsymp;
        a0, a, c, d = self.zconst, self.zscale, self.ylinscale, self.yquadscale;
        move = self.moveLimit;

        epsimin = 1e-7
        raa0 = 1e-5
        albefa = 0.1
        asyinit = 0.5
        asyincr = 1.2
        asydecr = 0.7
        eeen = np.ones((n, 1), dtype=np.float64)
        eeem = np.ones((m, 1), dtype=np.float64)
        zeron = np.zeros((n, 1), dtype=np.float64)

        # 计算渐近线low和upp
        if iter <= 2:
            low = xval - asyinit * (xmax - xmin)
            upp = xval + asyinit * (xmax - xmin)
        else:
            zzz = (xval - xold1) * (xold1 - xold2)
            factor = eeen.copy()
            factor[zzz > 0] = asyincr
            factor[zzz < 0] = asydecr
            low = xval - factor * (xold1 - low)
            upp = xval + factor * (upp - xold1)
            lowmin = xval - 10 * (xmax - xmin)
            lowmax = xval - 0.01 * (xmax - xmin)
            uppmin = xval + 0.01 * (xmax - xmin)
            uppmax = xval + 10 * (xmax - xmin)
            low = np.maximum(low, lowmin)
            low = np.minimum(low, lowmax)
            upp = np.minimum(upp, uppmax)
            upp = np.maximum(upp, uppmin)

        # 计算边界alfa和beta
        zzz1 = low + albefa * (xval - low)
        zzz2 = xval - move * (xmax - xmin)
        zzz = np.maximum(zzz1, zzz2)
        alfa = np.maximum(zzz, xmin)
        zzz1 = upp - albefa * (upp - xval)
        zzz2 = xval + move * (xmax - xmin)
        zzz = np.minimum(zzz1, zzz2)
        beta = np.minimum(zzz, xmax)

        # 计算p0, q0, P, Q和b
        xmami = xmax - xmin
        xmamieps = 1e-5 * eeen
        xmami = np.maximum(xmami, xmamieps)
        xmamiinv = eeen / xmami
        ux1 = upp - xval
        ux2 = ux1 * ux1
        xl1 = xval - low
        xl2 = xl1 * xl1
        uxinv = eeen / ux1
        xlinv = eeen / xl1

        p0 = zeron.copy()
        q0 = zeron.copy()
        p0 = np.maximum(df0dx, 0)
        q0 = np.maximum(-df0dx, 0)
        pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
        p0 = p0 + pq0
        q0 = q0 + pq0
        p0 = p0 * ux2
        q0 = q0 * xl2

        P = np.zeros((m, n), dtype=np.float64)
        Q = np.zeros((m, n), dtype=np.float64)
        P = np.maximum(dfdx, 0)
        Q = np.maximum(-dfdx, 0)
        PQ = 0.001 * (P + Q) + raa0 * np.dot(eeem, xmamiinv.T)
        P = P + PQ
        Q = Q + PQ

        # 替换JAX的数组乘法为NumPy乘法
        P = ux2.T * P
        Q = xl2.T * Q

        b = (np.dot(P, uxinv) + np.dot(Q, xlinv) - fval)

        # 调用子问题求解器
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
            m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)

        # 更新属性
        self.xmma, self.ymma, self.zmma = xmma, ymma, zmma;
        self.lam, self.xsi, self.eta, self.mu, self.zet = lam, xsi, eta, mu, zet;
        self.slack = s;
        self.lowAsymp, self.upAsymp = low, upp;


def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    een = np.ones((n, 1), dtype=np.float64)
    eem = np.ones((m, 1), dtype=np.float64)
    epsi = 1.0
    epsvecn = epsi * een
    epsvecm = epsi * eem
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = np.array([[1.0]], dtype=np.float64)
    lam = eem.copy()
    xsi = een / (x - alfa)
    xsi = np.maximum(xsi, een)
    eta = een / (beta - x)
    eta = np.maximum(eta, een)
    mu = np.maximum(eem, 0.5 * c)
    zet = np.array([[1.0]], dtype=np.float64)
    s = eem.copy()
    itera = 0

    # 数值稳定性循环
    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1
        plam = p0 + np.dot(P.T, lam)
        qlam = q0 + np.dot(Q.T, lam)
        gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - np.dot(a.T, lam)
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm
        residu1 = np.concatenate((rex, rey, rez), axis=0)
        residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
        residu = np.concatenate((residu1, residu2), axis=0)
        residunorm = np.sqrt((np.dot(residu.T, residu)).item())
        residumax = np.max(np.abs(residu))
        ittt = 0

        # 内循环优化
        while (residumax > 0.9 * epsi) and (ittt < 200):
            ittt += 1
            itera += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2
            plam = p0 + np.dot(P.T, lam)
            qlam = q0 + np.dot(Q.T, lam)
            gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)

            # 替换JAX的稀疏矩阵操作，使用NumPy数组运算
            GG = uxinv2.T * P - xlinv2.T * Q

            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - np.dot(a.T, lam) - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            # 求解方程组
            if m < n:
                blam = dellam + dely / diagy - np.dot(GG, (delx / diagx))
                bb = np.concatenate((blam, delz), axis=0)

                # 构造系数矩阵（使用NumPy对角矩阵）
                Alam = diags(diaglamyi.flatten(), 0) + (diagxinv.T * GG).dot(GG.T)
                AAr1 = np.concatenate((Alam, a), axis=1)
                AAr2 = np.concatenate((a, -zet / z), axis=0).T
                AA = np.concatenate((AAr1, AAr2), axis=0)
                solut = solve(AA, bb)
                dlam = solut[0:m]
                dz = solut[m:m + 1]
                dx = -delx / diagx - np.dot(GG.T, dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = diags(diagx.flatten(), 0) + (diags(diaglamyiinv.flatten(), 0).dot(GG.T)).dot(GG)
                azz = zet / z + np.dot(a.T, (a / diaglamyi))
                axz = np.dot(-GG.T, (a / diaglamyi))
                bx = delx + np.dot(GG.T, (dellamyi / diaglamyi))
                bz = delz - np.dot(a.T, (dellamyi / diaglamyi))
                AAr1 = np.concatenate((Axx, axz), axis=1)
                AAr2 = np.concatenate((axz.T, azz), axis=1)
                AA = np.concatenate((AAr1, AAr2), axis=0)
                bb = np.concatenate((-bx, -bz), axis=0)
                solut = solve(AA, bb)
                dx = solut[0:n]
                dz = solut[n:n + 1]
                dlam = np.dot(GG, dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam
            xx = np.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)
            dxx = np.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0)

            # 步长确定
            stepxx = -1.01 * dxx / xx
            stmxx = np.max(stepxx)
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0 / stminv

            # 更新变量
            xold = x.copy()
            yold = y.copy()
            zold = z.copy()
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet.copy()
            sold = s.copy()

            itto = 0
            resinew = 2 * residunorm

            # 线搜索
            while (resinew > residunorm) and (itto < 50):
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = een / ux1
                xlinv1 = een / xl1
                plam = p0 + np.dot(P.T, lam)
                qlam = q0 + np.dot(Q.T, lam)
                gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
                dpsidx = plam / ux2 - qlam / xl2
                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - np.dot(a.T, lam)
                relam = gvec - np.dot(a, z) - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = np.dot(zet, z) - epsi
                res = lam * s - epsvecm
                residu1 = np.concatenate((rex, rey, rez), axis=0)
                residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
                residu = np.concatenate((residu1, residu2), axis=0)
                resinew = np.sqrt(np.dot(residu.T, residu))
                steg = steg / 2

            residunorm = resinew.copy()
            residumax = max(abs(residu))
            steg = 2 * steg

        epsi = 0.1 * epsi

    xmma = x.copy()
    ymma = y.copy()
    zmma = z.copy()
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma


def optimize(fe, p_ini, optimizationParams, objectiveHandle, consHandle, numConstraints):
    # 计算过滤矩阵（使用SciPy稀疏矩阵）
    if density_filtering or sensitivity_filtering:
        H, Hs = compute_filter_kd_tree(fe)
        ft = {'H': H, 'Hs': Hs}

    p = p_ini.astype(np.float64)  # 确保64位精度

    loop = 0
    m = numConstraints  # 约束数量
    n = len(p.reshape(-1))  # 设计变量数量

    mma = MMA()
    mma.setNumConstraints(numConstraints)
    mma.setNumDesignVariables(n)
    mma.setMinandMaxBoundsForDesignVariables(
        np.zeros((n, 1), dtype=np.float64),
        np.ones((n, 1), dtype=np.float64)
    )

    xval = p.reshape(-1)[:, None].astype(np.float64)
    xold1, xold2 = xval.copy(), xval.copy()
    mma.registerMMAIter(xval, xold1, xold2)
    mma.setLowerAndUpperAsymptotes(
        np.ones((n, 1), dtype=np.float64),
        np.ones((n, 1), dtype=np.float64)
    )
    mma.setScalingParams(
        1.0,
        np.zeros((m, 1), dtype=np.float64),
        10000 * np.ones((m, 1), dtype=np.float64),
        np.zeros((m, 1), dtype=np.float64)
    )
    mma.setMoveLimit(optimizationParams['movelimit'])

    while loop < optimizationParams['maxIters']:
        loop += 1

        print(f"MMA solver...")

        # 应用密度过滤
        if density_filtering:
            p_physical = applyDensityFilter(ft, p)
        else:
            p_physical = p

        # 计算目标函数和约束（用户提供的接口）
        J, dJ = objectiveHandle(p_physical)
        vc, dvc = consHandle(p_physical, loop)

        # 应用灵敏度过滤
        if sensitivity_filtering:
            dJ, dvc = applySensitivityFilter(ft, p, dJ, dvc)

        # 调整形状为列向量
        J, dJ = J, dJ.reshape(-1)[:, None].astype(np.float64)
        vc, dvc = vc[:, None].astype(np.float64), dvc.reshape(dvc.shape[0], -1).astype(np.float64)

        print(f"J.shape = {J.shape}")
        print(f"dJ.shape = {dJ.shape}")
        print(f"vc.shape = {vc.shape}")
        print(f"dvc.shape = {dvc.shape}")

        start = time.time()

        # 更新MMA优化器状态并求解
        mma.setObjectiveWithGradient(J, dJ)
        mma.setConstraintWithGradient(vc, dvc)
        mma.mmasub(xval)
        xmma, _, _ = mma.getOptimalValues()

        # 更新迭代历史
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()

        mma.registerMMAIter(xval, xold1, xold2)
        p = xval.reshape(p.shape)

        end = time.time()
        time_elapsed = end - start

        print(f"MMA took {time_elapsed:.4f} [s]")
        print(f'Iter {loop:d}; J {J.item():.5f}; constraint {vc.flatten()}\n')

    return p
