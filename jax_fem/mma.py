"""
Copied and modified from https://github.com/UW-ERSL/AuTO
Under GNU General Public License v3.0

Original copy from https://github.com/arjendeetman/GCMMA-MMA-Python/blob/master/Code/MMA.py

Improvement is made to avoid N^2 memory operation so that the MMA solver is more scalable.
"""
from numpy import diag as diags
from numpy.linalg import solve
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, random, jacfwd, value_and_grad
from functools import partial
import time
import scipy
from jax.experimental.sparse import BCOO
from typing import Callable
from jax import config

from jax_fem.utils import Jplotter
config.update("jax_enable_x64", True)



def compute_filter_kd_tree(fe,r_factor=1.5):
    """This function is created by Tianju. Not from the original code.
    We use k-d tree algorithm to compute the filter.
    r_factor <1: nothing but itself included,
    r_factor =1: direct nearby cells included,
    r_factor = sqrt(3) surround included,
    r_factor >2 more included.
    """
    cell_centroids = np.mean(np.take(fe.points, fe.cells, axis=0), axis=1)
    flex_num_cells = len(fe.flex_inds)
    flex_cell_centroids = np.take(cell_centroids, fe.flex_inds, axis=0)

    V = np.sum(fe.JxW)
    avg_elem_V = V/fe.num_cells

    avg_elem_size = avg_elem_V**(1./fe.dim)
    rmin = r_factor*avg_elem_size

    kd_tree = scipy.spatial.KDTree(flex_cell_centroids)
    I = []
    J = []
    V = []
    for i in range(flex_num_cells):
        num_nbs = 20
        dd, ii = kd_tree.query(flex_cell_centroids[i], num_nbs)
        neighbors = np.take(flex_cell_centroids, ii, axis=0)
        vals = np.where(rmin - dd > 0., rmin - dd, 0.)
        I += [i]*num_nbs
        J += ii.tolist()
        V += vals.tolist()
    H_sp = scipy.sparse.csc_array((V, (I, J)), shape=(flex_num_cells, flex_num_cells))

    H = BCOO.from_scipy_sparse(H_sp).sort_indices()
    Hs = H.sum(1).todense()
    return H, Hs

def applySensitivityFilter(ft, rho, dJ, dvc):
    dJ = ft['H'] @ (rho*dJ/np.maximum(1e-3, rho)/ft['Hs'][:, None]) # (n, n) @ [(n,1) * (n,1)/(1,)/(1, 1) ]-> (n,n)@(n, 1)->(n, 1)
    dvc = ft['H'][None, :, :] @ (rho[None, :, :]*dvc/np.maximum(1e-3, rho[None, :, :])/ft['Hs'][None, :, None]) # (1, n, n)@(1, n, 1) -> (1, n, 1)
    return dJ, dvc

def applyDensityFilter(ft, rho):
    return ft['H'] @ rho / ft['Hs'][:, None]


def applySensitivityFilter_multi(ft, rho, dJ, dvc, beta=1.0):
    """
    修正版多材料灵敏度过滤器：兼容多约束(C, N, n)，使用JAX规范
    dJ: 形状(N, n)，dvc: 形状(C, N, n)（C为约束数量，支持C=1或C>1）
    """
    N, n = rho.shape
    H = ft['H']  # JAX BCOO稀疏矩阵 (N, N)
    
    # 1. 计算总材料量和其他材料存在度（与密度过滤一致）
    total_material = rho.sum(axis=1, keepdims=True)  # (N, 1)
    O_jk = total_material - rho  # (N, n)：其他材料存在度
    weight_factor = 1 + beta * O_jk  # (N, n)：耦合权重因子
    weight_sum = H @ weight_factor  # (N, n)：归一化因子（共享）
    
    # ---------------------- 处理目标函数灵敏度dJ ----------------------
    pre_dJ = (rho * dJ) / jnp.maximum(1e-3, rho)  # JAX函数，(N, n)
    weighted_dJ = pre_dJ * weight_factor  # (N, n)
    numerator_dJ = H @ weighted_dJ  # (N, n)
    dJ_tilde = numerator_dJ / jnp.maximum(1e-12, weight_sum)  # (N, n)
    
    # ---------------------- 处理体积约束灵敏度dvc（兼容多约束） ----------------------
    # 扩展rho和weight_factor的维度，与dvc的(C, N, n)广播匹配
    rho_expanded = rho[jnp.newaxis, ...]  # (1, N, n) → 与dvc的(C, N, n)广播
    weight_factor_expanded = weight_factor[jnp.newaxis, ...]  # (1, N, n)
    
    # 灵敏度预处理（保留约束维度C）
    pre_dvc = (rho_expanded * dvc) / jnp.maximum(1e-3, rho_expanded)  # (C, N, n)
    
    # 加权（耦合因子作用于所有约束）
    weighted_dvc = pre_dvc * weight_factor_expanded  # (C, N, n)
    
    # 稀疏矩阵批量乘法（H仅作用于单元维度N）
    numerator_dvc = H @ weighted_dvc  # (C, N, n)
    
    # 归一化（保留约束维度）
    dvc_tilde = numerator_dvc / jnp.maximum(1e-12, weight_sum[jnp.newaxis, ...])  # (C, N, n)
    
    return dJ_tilde, dvc_tilde


def applyDensityFilter_multi(ft, rho, beta=1.0):
    """
    多材料密度过滤器（向量计算版）：同时处理所有材料，避免循环
    输入rho形状为(N, n)，N为单元数，n为材料种类
    """
    N, n = rho.shape
    H = ft['H']  # 稀疏矩阵 (N, N)，单元邻域权重
    
    # 1. 计算总材料量（每个单元所有材料的总和）：形状(N, 1)
    total_material = rho.sum(axis=1, keepdims=True)  # 广播到(N, n)
    
    # 2. 计算其他材料存在度 O_j^k = 总材料量 - 当前材料量：形状(N, n)
    O_jk = total_material - rho  # 利用广播，自动适配每个材料k
    
    # 3. 计算加权因子：(1 + beta * O_j^k)，形状(N, n)
    weight_factor = 1 + beta * O_jk
    
    # 4. 计算分子项：H @ (rho * weight_factor)，形状(N, n)
    # 稀疏矩阵H与(N, n)数组相乘时，自动对每一列做矩阵-向量乘法
    numerator = H @ (rho * weight_factor)  # 结果(N, n)
    
    # 5. 计算归一化因子：H @ weight_factor，形状(N, n)
    weight_sum = H @ weight_factor  # 结果(N, n)
    
    # 6. 过滤后密度（避免除零）
    rho_tilde = numerator / np.maximum(1e-12, weight_sum)
    
    return rho_tilde


#%% Optimizer
class MMA:
    # The code was modified from [MMA Svanberg 1987]. Please cite the paper if
    # you end up using this code.
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

    # Function for the MMA sub problem
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

        epsimin = 0.0000001
        raa0 = 0.00001
        albefa = 0.1
        asyinit = 0.5
        asyincr = 1.2
        asydecr = 0.7
        eeen = np.ones((n, 1))
        eeem = np.ones((m, 1))
        zeron = np.zeros((n, 1))
        # Calculation of the asymptotes low and upp
        if iter <= 2:
            low = xval-asyinit*(xmax-xmin)
            upp = xval+asyinit*(xmax-xmin)
        else:
            zzz = (xval-xold1)*(xold1-xold2)
            factor = eeen.copy()
            factor[np.where(zzz>0)] = asyincr
            factor[np.where(zzz<0)] = asydecr
            low = xval-factor*(xold1-low)
            upp = xval+factor*(upp-xold1)
            lowmin = xval-10*(xmax-xmin)
            lowmax = xval-0.01*(xmax-xmin)
            uppmin = xval+0.01*(xmax-xmin)
            uppmax = xval+10*(xmax-xmin)
            low = np.maximum(low,lowmin)
            low = np.minimum(low,lowmax)
            upp = np.minimum(upp,uppmax)
            upp = np.maximum(upp,uppmin)
        # Calculation of the bounds alfa and beta
        zzz1 = low+albefa*(xval-low)
        zzz2 = xval-move*(xmax-xmin)
        zzz = np.maximum(zzz1,zzz2)
        alfa = np.maximum(zzz,xmin)
        zzz1 = upp-albefa*(upp-xval)
        zzz2 = xval+move*(xmax-xmin)
        zzz = np.minimum(zzz1,zzz2)
        beta = np.minimum(zzz,xmax)
        # Calculations of p0, q0, P, Q and b
        xmami = xmax-xmin
        xmamieps = 0.00001*eeen
        xmami = np.maximum(xmami,xmamieps)
        xmamiinv = eeen/xmami
        ux1 = upp-xval
        ux2 = ux1*ux1
        xl1 = xval-low
        xl2 = xl1*xl1
        uxinv = eeen/ux1
        xlinv = eeen/xl1
        p0 = zeron.copy()
        q0 = zeron.copy()
        p0 = np.maximum(df0dx,0)
        q0 = np.maximum(-df0dx,0)
        pq0 = 0.001*(p0+q0)+raa0*xmamiinv
        p0 = p0+pq0
        q0 = q0+pq0
        p0 = p0*ux2
        q0 = q0*xl2
        P = np.zeros((m,n)) ## @@ make sparse with scipy?
        Q = np.zeros((m,n)) ## @@ make sparse with scipy?
        P = np.maximum(dfdx,0)
        Q = np.maximum(-dfdx,0)
        PQ = 0.001*(P+Q)+raa0*np.dot(eeem,xmamiinv.T)
        P = P+PQ
        Q = Q+PQ

        # P = (diags(ux2.flatten(),0).dot(P.T)).T
        # Q = (diags(xl2.flatten(),0).dot(Q.T)).T
        P = ux2.T*P
        Q = xl2.T*Q

        b = (np.dot(P,uxinv)+np.dot(Q,xlinv)-fval)
        # Solving the subproblem by a primal-dual Newton method
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s = subsolv(m,n,epsimin,low,upp,alfa,\
                                                      beta,p0,q0,P,Q,a0,a,b,c,d)
        # Return values
        self.xmma, self.ymma, self.zmma = xmma, ymma, zmma;
        self.lam, self.xsi, self.eta, self.mu, self.zet = lam,xsi,eta,mu,zet;
        self.slack = s;
        self.lowAsymp, self.upAsymp = low, upp;


def subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d):
    een = np.ones((n,1))
    eem = np.ones((m,1))
    epsi = 1
    epsvecn = epsi*een
    epsvecm = epsi*eem
    x = 0.5*(alfa+beta)
    y = eem.copy()
    z = np.array([[1.0]])
    lam = eem.copy()
    xsi = een/(x-alfa)
    xsi = np.maximum(xsi,een)
    eta = een/(beta-x)
    eta = np.maximum(eta,een)
    mu = np.maximum(eem,0.5*c)
    zet = np.array([[1.0]])
    s = eem.copy()
    itera = 0
    # Start while epsi>epsimin
    while epsi > epsimin:
        epsvecn = epsi*een
        epsvecm = epsi*eem
        ux1 = upp-x
        xl1 = x-low
        ux2 = ux1*ux1
        xl2 = xl1*xl1
        uxinv1 = een/ux1
        xlinv1 = een/xl1
        plam = p0+np.dot(P.T,lam)
        qlam = q0+np.dot(Q.T,lam)
        gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1)
        dpsidx = plam/ux2-qlam/xl2
        rex = dpsidx-xsi+eta
        rey = c+d*y-mu-lam
        rez = a0-zet-np.dot(a.T,lam)
        relam = gvec-a*z-y+s-b
        rexsi = xsi*(x-alfa)-epsvecn
        reeta = eta*(beta-x)-epsvecn
        remu = mu*y-epsvecm
        rezet = zet*z-epsi
        res = lam*s-epsvecm
        residu1 = np.concatenate((rex, rey, rez), axis = 0)
        residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis = 0)
        residu = np.concatenate((residu1, residu2), axis = 0)
        residunorm = np.sqrt((np.dot(residu.T,residu)).item())
        residumax = np.max(np.abs(residu))
        ittt = 0
        # Start while (residumax>0.9*epsi) and (ittt<200)
        while (residumax > 0.9*epsi) and (ittt < 200):
            ittt = ittt+1
            itera = itera+1
            ux1 = upp-x
            xl1 = x-low
            ux2 = ux1*ux1
            xl2 = xl1*xl1
            ux3 = ux1*ux2
            xl3 = xl1*xl2
            uxinv1 = een/ux1
            xlinv1 = een/xl1
            uxinv2 = een/ux2
            xlinv2 = een/xl2
            plam = p0+np.dot(P.T,lam)
            qlam = q0+np.dot(Q.T,lam)
            gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1)

            # GG = (diags(uxinv2.flatten(),0).dot(P.T)).T-(diags\
            #                          (xlinv2.flatten(),0).dot(Q.T)).T
            GG = uxinv2.T*P - xlinv2.T*Q

            dpsidx = plam/ux2-qlam/xl2
            delx = dpsidx-epsvecn/(x-alfa)+epsvecn/(beta-x)
            dely = c+d*y-lam-epsvecm/y
            delz = a0-np.dot(a.T,lam)-epsi/z
            dellam = gvec-a*z-y-b+epsvecm/lam
            diagx = plam/ux3+qlam/xl3
            diagx = 2*diagx+xsi/(x-alfa)+eta/(beta-x)
            diagxinv = een/diagx
            diagy = d+mu/y
            diagyinv = eem/diagy
            diaglam = s/lam
            diaglamyi = diaglam+diagyinv
            # Start if m<n
            if m < n:
                blam = dellam+dely/diagy-np.dot(GG,(delx/diagx))
                bb = np.concatenate((blam,delz),axis = 0)
                
                # Alam = np.asarray(diags(diaglamyi.flatten(),0) \
                #     +(diags(diagxinv.flatten(),0).dot(GG.T).T).dot(GG.T))
                Alam = diags(diaglamyi.flatten(),0) + (diagxinv.T*GG).dot(GG.T)

                AAr1 = np.concatenate((Alam,a),axis = 1)
                AAr2 = np.concatenate((a,-zet/z),axis = 0).T
                AA = np.concatenate((AAr1,AAr2),axis = 0)
                solut = solve(AA,bb)
                dlam = solut[0:m]
                dz = solut[m:m+1]
                dx = -delx/diagx-np.dot(GG.T,dlam)/diagx
            else:
                diaglamyiinv = eem/diaglamyi
                dellamyi = dellam+dely/diagy
                Axx = np.asarray(diags(diagx.flatten(),0) \
                    +(diags(diaglamyiinv.flatten(),0).dot(GG).T).dot(GG))
                azz = zet/z+np.dot(a.T,(a/diaglamyi))
                axz = np.dot(-GG.T,(a/diaglamyi))
                bx = delx+np.dot(GG.T,(dellamyi/diaglamyi))
                bz = delz-np.dot(a.T,(dellamyi/diaglamyi))
                AAr1 = np.concatenate((Axx,axz),axis = 1)
                AAr2 = np.concatenate((axz.T,azz),axis = 1)
                AA = np.concatenate((AAr1,AAr2),axis = 0)
                bb = np.concatenate((-bx,-bz),axis = 0)
                solut = solve(AA,bb)
                dx = solut[0:n]
                dz = solut[n:n+1]
                dlam = np.dot(GG,dx)/diaglamyi-dz*(a/diaglamyi)\
                    +dellamyi/diaglamyi
                # End if m<n
            dy = -dely/diagy+dlam/diagy
            dxsi = -xsi+epsvecn/(x-alfa)-(xsi*dx)/(x-alfa)
            deta = -eta+epsvecn/(beta-x)+(eta*dx)/(beta-x)
            dmu = -mu+epsvecm/y-(mu*dy)/y
            dzet = -zet+epsi/z-zet*dz/z
            ds = -s+epsvecm/lam-(s*dlam)/lam
            xx = np.concatenate((y,z,lam,xsi,eta,mu,zet,s),axis = 0)
            dxx = np.concatenate((dy,dz,dlam,dxsi,deta,dmu,dzet,ds),axis = 0)
            #
            stepxx = -1.01*dxx/xx
            stmxx = np.max(stepxx)
            stepalfa = -1.01*dx/(x-alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01*dx/(beta-x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa,stmbeta)
            stmalbexx = max(stmalbe,stmxx)
            stminv = max(stmalbexx,1.0)
            steg = 1.0/stminv
            #
            xold = x.copy()
            yold = y.copy()
            zold = z.copy()
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet.copy()
            sold = s.copy()
            #
            itto = 0
            resinew = 2*residunorm

            # Start: while (resinew>residunorm) and (itto<50)
            while (resinew > residunorm) and (itto < 50):
                itto = itto+1
                x = xold+steg*dx
                y = yold+steg*dy
                z = zold+steg*dz
                lam = lamold+steg*dlam
                xsi = xsiold+steg*dxsi
                eta = etaold+steg*deta
                mu = muold+steg*dmu
                zet = zetold+steg*dzet
                s = sold+steg*ds
                ux1 = upp-x
                xl1 = x-low
                ux2 = ux1*ux1
                xl2 = xl1*xl1
                uxinv1 = een/ux1
                xlinv1 = een/xl1
                plam = p0+np.dot(P.T,lam)
                qlam = q0+np.dot(Q.T,lam)
                gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1)
                dpsidx = plam/ux2-qlam/xl2
                rex = dpsidx-xsi+eta
                rey = c+d*y-mu-lam
                rez = a0-zet-np.dot(a.T,lam)
                relam = gvec-np.dot(a,z)-y+s-b
                rexsi = xsi*(x-alfa)-epsvecn
                reeta = eta*(beta-x)-epsvecn
                remu = mu*y-epsvecm
                rezet = np.dot(zet,z)-epsi
                res = lam*s-epsvecm
                residu1 = np.concatenate((rex,rey,rez),axis = 0)
                residu2 = np.concatenate((relam,rexsi,reeta,remu,rezet,res), \
                                         axis = 0)
                residu = np.concatenate((residu1,residu2),axis = 0)
                resinew = np.sqrt(np.dot(residu.T,residu))
                steg = steg/2
                # End: while (resinew>residunorm) and (itto<50)

            residunorm = resinew.copy()
            residumax = max(abs(residu))
            steg = 2*steg
            # End: while (residumax>0.9*epsi) and (ittt<200)
        epsi = 0.1*epsi
        # End: while epsi>epsimin

    xmma = x.copy()
    ymma = y.copy()
    zmma = z.copy()
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma,ymma,zmma,lamma,xsimma,etamma,mumma,zetmma,smma


def optimize(fe, rho_ini, optimizationParams, objectiveHandle, consHandle, numConstraints,tileNum, WFC:Callable):
    """
    Performs topology optimization using the Method of Moving Asymptotes (MMA).

    Parameters
    ----------
    fe : FiniteElement
        Finite element object.
    rho_ini : NumpyArray
        Initial density distribution.
        Shape is (num_rho_vars, n).
    optimizationParams : dict
        Dictionary containing optimization parameters:

        - 'movelimit': Move limit for design variables (float)
        - 'maxIters': Maximum number of iterations (int)
    objectiveHandle : callable
        Function that computes the objective value and its gradient.
        Signature: ``J, dJ = objectiveHandle(rho_physical)``

        - ``rho_physical``: Physical density field (filtered if enabled). Same shape as ``rho_ini``.
        - ``J``: Objective value (scalar).
        - ``dJ``: Objective gradient (NumpyArray, same shape as ``rho_ini``).
    consHandle : callable
        Function that computes constraint values and their gradients.
        Signature: ``vc, dvc = consHandle(rho_physical, iter)``

        - ``rho_physical``: Physical density field (filtered if enabled). Same shape as ``rho_ini``.
        - ``iter``: Current optimization iteration (int).
        - ``vc``: Constraint values. Shape is (num_constraints,).
        - ``dvc``: Constraint gradients. Shape is (num_constraints, ...), where ... shares the same shape with ``rho_ini``.
    numConstraints : int
        Number of constraints in the optimization problem.

    Returns
    -------
    rho : NumpyArray
        Optimized density distribution after completing all iterations.
        Same shape as ``rho_ini``.

    Notes
    -----
    TODO: Scale objective function value to be always within 1-100
    (`ref <https://doi.org/10.1016/j.compstruc.2018.01.008>`_).
    """
    # stop condition
    tol_obj = optimizationParams.get('tol_obj', 1e-5)
    tol_design = optimizationParams.get('tol_design', 1e-1)
    tol_con = optimizationParams.get('tol_con', 1e-1)
    min_iters = optimizationParams.get('min_iters', 10)
    sensitivity_filtering = optimizationParams.get('sensitivity_filtering',"common")


    # jplotter = Jplotter()
    J_prev = np.inf
    rho_prev = rho_ini.copy()
    rho = rho_ini
    infos={}
    con_violation_last = 0
    rfmean_last=0
    rfmin_last=0
    rfmax_last=0

    H, Hs = compute_filter_kd_tree(fe,r_factor = optimizationParams.get('filter_radius', 1.8)) #1.8
    ft = {'H':H, 'Hs':Hs}

    loop = 0
    m = numConstraints # num constraints
    n = len(rho.reshape(-1)) # num params

    mma = MMA()
    mma.setNumConstraints(numConstraints)
    mma.setNumDesignVariables(n)
    mma.setMinandMaxBoundsForDesignVariables\
        (np.zeros((n,1)),np.ones((n,1)))

    xval = rho.reshape(-1)[:, None] 
    xold1, xold2 = xval.copy(), xval.copy()
    mma.registerMMAIter(xval, xold1, xold2)
    mma.setLowerAndUpperAsymptotes(np.ones((n,1)), np.ones((n,1)))
    mma.setScalingParams(1.0, np.zeros((m,1)), \
                         10000*np.ones((m,1)), np.zeros((m,1)))
    # Move limit is an important parameter that affects TO result; default can be 0.2
    mma.setMoveLimit(optimizationParams['movelimit']) 
    allstart = time.time()

    while loop < optimizationParams['maxIters']:
        start_time=time.time()
        loop = loop + 1
        info={}
        np.save(f"data/npy/{loop}",rho)
        # alpha = 0.2 + 0.6 / (1 + np.exp(-10 * (loop / optimizationParams['maxIters'] - 0.5))) #0.2-0.8, 10越大越陡峭
        alpha = 1
        
        print(f"MMA solver...")
        print(f"collapsing...")
        print(f"rho.shape: {rho.shape}")
        print("rho 均值：", jnp.mean(rho))
        print("rho 最小值：", jnp.min(rho))
        print("rho 最大值：", jnp.max(rho))

        def filter_chain(rho,WFC,ft,loop):
            # rho = applyDensityFilter(ft, rho)
            rho,_,_=WFC(rho.reshape(-1,tileNum))
            rho = rho.reshape(-1,tileNum) #不一定需要reshaped到(...,1)
            # rho = jax.nn.softmax(rho,axis=-1)
            # rho = heaviside(rho,2^(loop//10))
            # rho = smooth_heaviside(rho, beta=2**((loop-0.5)//5))
            return rho
        # 2. 对filter_chain构建VJP（关键：函数依赖输入r）
        def filter_chain_vjp(r):
            return filter_chain(r, WFC, ft,loop)

        # 构建VJP：fwd_func返回(rho_f, vjp_fn)，其中vjp_fn用于计算梯度
        rho_f_vjp, vjp_fn = jax.vjp(filter_chain_vjp, rho)


        rho_f = rho_f_vjp
        rfmean = jnp.mean(rho_f)
        rfmin = jnp.min(rho_f)
        rfmax = jnp.max(rho_f)
        print(f"rho_f 均值：{rfmean}; change:{rfmean - rfmean_last}")
        print(f"rho_f 最小值：{rfmin}; change:{rfmin - rfmin_last}")
        print(f"rho_f 最大值：{rfmax}; change:{rfmax - rfmax_last}")
        rfmean_last = rfmean
        rfmin_last = rfmin
        rfmax_last = rfmax

        J, dJ_drho_f = objectiveHandle(rho_f)  # dJ_drho_f：目标函数对rho_f的梯度
        vc, dvc_drho_f = consHandle(rho_f)     # dvc_drho_f：约束对rho_f的梯度
        print(f"dJ_drho_f.shape: {dJ_drho_f.shape}\ndvc_drho_f.shape: {dvc_drho_f.shape}")
        print(f"dJ_drho_f.max: {jnp.max(dJ_drho_f)}")
        print(f"dJ_drho_f.min: {jnp.min(dJ_drho_f)}")
        print(f"dvc_drho_f.max: {jnp.max(dvc_drho_f)}")
        print(f"dvc_drho_f.min: {jnp.min(dvc_drho_f)}")

        # # 关键：用vjp_fn计算rho对rho_f的梯度，再乘以dJ_drho_f（链式法则）
        dJ_drho = vjp_fn(dJ_drho_f)[0]
        vjp_batch = jax.vmap(vjp_fn, in_axes=0, out_axes=0)
        dvc_drho = vjp_batch(dvc_drho_f)[0]
        print(f"dJ_drho.shape: {dJ_drho.shape}\ndvc_drho.shape: {dvc_drho.shape}")
        print(f"dJ_drho.max: {jnp.max(dJ_drho)}")   
        print(f"dJ_drho.min: {jnp.min(dJ_drho)}")
        print(f"dvc_drho.max: {jnp.max(dvc_drho)}")
        print(f"dvc_drho.min: {jnp.min(dvc_drho)}")

        dJ=dJ_drho
        dvc=dvc_drho
        print(f"sensitivity filtering: {sensitivity_filtering}")
        if sensitivity_filtering=="common":
            dJ, dvc = applySensitivityFilter(ft, rho_f, dJ, dvc) #一直用的这个做++TT0TT180 完全约束f1 f1.5 p544 p444
            print(f"sensitivity filtering: common")

        if sensitivity_filtering=="multi":
            dJ, dvc = applySensitivityFilter_multi(ft, rho_f, dJ, dvc,beta=1.0)
            print(f"sensitivity filtering: multi")

        

        print(f"dJ.shape: {dJ.shape}\ndvc.shape: {dvc.shape}")


        J, dJ = J, dJ.reshape(-1)[:, None]
        vc, dvc = vc[:, None], dvc.reshape(dvc.shape[0], -1)

        print(f"dJ.max: {np.max(dJ)}")
        print(f"dJ.min: {np.min(dJ)}")


        J, dJ, vc, dvc = np.array(J), np.array(dJ), np.array(vc), np.array(dvc)
        # jplotter.update(loop, J)

        start = time.time()

        mma.setObjectiveWithGradient(J, dJ)
        mma.setConstraintWithGradient(vc, dvc)
        mma.mmasub(xval)
        xmma, _, _ = mma.getOptimalValues()

        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        # 目标函数变化
        obj_change = J - J_prev
        
        # # 设计变量变化
        # design_change = np.linalg.norm(rho - rho_prev)
        
        # 约束满足度
        con_violation = np.max(vc)
        
        #灰度监控
        grayness, clear_ratio, _ = compute_material_grayness(rho_f)
        info.update({"grayness":grayness,"clear_ratio":clear_ratio,"con_violation":con_violation,"J":J})
        # 梯度范数
        print("****************************************************")
        print(f"{optimizationParams}")
        print(f"J: {J}")
        print(f"obj_change:{obj_change}; tol:{max(1e-6, tol_obj*abs(J_prev))}")
        print(f"constrain violation:{con_violation}; tol:{tol_con}")
        print(f"constrain change:{con_violation-con_violation_last}")
        print(f"Grayness: {grayness:.4f} | Clear: {clear_ratio:.1%}")
        
        if loop > min_iters:
            if abs(obj_change) < max(1e-6, tol_obj*abs(J_prev)) and\
                (con_violation <= tol_con):
                print(f"收敛于迭代 {loop}")
                break


        mma.registerMMAIter(xval, xold1, xold2)
        rho = xval.reshape(rho.shape)
        rho = np.clip(rho, 0, 1)
        end = time.time()

        time_elapsed = end - start

        print(f"MMA took {time_elapsed} [s]")

        print(f'Iter {loop:d} end; J {J:.5f}; \nconstraint: \n{vc}')
        print(f"epoch spends: {time.time()-start_time} [s]")
        print("****************************************************\n")
        J_prev = J
        rho_prev = rho.copy()
        con_violation_last = con_violation
        infos[loop]=info
    # jplotter.finalize()
    print(f"Total optimization time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-allstart))} [s]")
    return rho,infos


def compute_material_grayness(rho_f: jnp.ndarray, 
                             threshold: float = 0.95):
    """
    即插即用的多材料灰度监控函数
    
    参数:
    rho_f: 概率场, 形状为 (num_elements, num_materials)
    threshold: 清晰度阈值, 默认0.95(95%概率视为清晰选择)
    
    返回:
    grayness: 平均灰度指标 (0-1之间, 越小越好)
    clear_ratio: 清晰单元比例 (0-1之间, 越大越好)  
    element_grayness: 每个单元的灰度值, 用于详细分析
    """
    # 数值稳定性: 确保概率和为1
    # rho_normalized = rho_f / (jnp.sum(rho_f, axis=-1, keepdims=True) + 1e-12)
    # 计算每个单元的最大概率 (清晰度)
    max_probs = jnp.max(rho_f, axis=-1)
    # 灰度 = 1 - 最大概率 (度量模糊程度)
    element_grayness = 1.0 - max_probs
    # 全局平均灰度
    grayness = jnp.mean(element_grayness)
    # 清晰单元比例 (最大概率超过阈值)
    clear_ratio = jnp.sum(max_probs > threshold)/ rho_f.shape[0]
    return float(grayness), float(clear_ratio), element_grayness

@jit
def heaviside(x: jnp.ndarray,eta=0.5, beta: float = 10.0) -> jnp.ndarray:
    """
    简化的Heaviside投影函数。
    
    参数:
        x: 输入数组
        beta: 投影锐度参数
    返回:
        投影后的数组
    """
    return jnp.tanh(beta * eta) + jnp.tanh(beta * (x - eta)) / (
        jnp.tanh(beta * eta) + jnp.tanh(beta * eta))


@partial(jit, static_argnames=('beta',))
def smooth_heaviside(x,beta=10.):
    """
    greater the beta, sharper the curve.
    """
    return 1/(1+jnp.exp(-beta*x))