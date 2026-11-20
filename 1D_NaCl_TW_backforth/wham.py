import copy
import os
import numpy as np
import jax
import functools
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from sklearn.cluster import KMeans


def sum_Fx(expFx_old, sim):
    denom = np.sum(expFx_old * sim)
    if denom > 0:
        denom = 1 / denom
    else:
        denom = 0
    return denom * sim


def rmsd(offset, a, b):
    return np.sqrt(np.mean(np.square((a + offset) - b)))


def align(query, ref):
    offset = -10.0
    res = minimize(rmsd, offset, args=(query, ref))
    print(res.x[0], res.fun)
    return res.x[0]


@jax.jit
def jax_mult(A, B):
    return A * B


@jax.jit
def jax_mult_broadcast(A, B):
    return A * B


def create_bins(q, numbins):
    v_min = np.min(q)
    v_max = np.max(q)
    return np.linspace(v_min, v_max, numbins)


def cnt_pop(qep, qspace, denom, numsims, numbins=50):
    b = np.digitize(qep, qspace) - 1
    # P = np.empty(shape=numbins)
    PpS = np.empty(shape=(numsims, numbins))
    for i in range(numbins):
        md = np.ma.masked_array(denom, mask=~(b == i))
        # P[i] = np.ma.sum(md)
        PpS[:, i] = np.ma.sum(md, axis=1)
    P = np.sum(PpS, axis=0)
    return P, PpS


class WHAM:
    """
    data is 3D: (number of sims, points per sims, number of colvars)
    kval and constr_val are 2D: (number of sims, number of colvars)
    """
    skip = 10
    KbT = 0.001987204259 * 300  # energy unit: kcal/mol
    data = None     # in A
    k_val = None    # in kcal/mol/A**2
    constr_val = None   # in A
    winsize = None
    UB = None
    Fprog = None
    denom = None

    def __init__(self, path):
        self.path = path
        return

    def read(self, strings=range(2, 9)):
        coor = None
        force = None
        data = None
        winsize = []
        for s in strings:
            # For string #s the constraints are in allconstr_{s-1}
            # the data is in coll_win{s}
            actcoor = np.loadtxt(os.path.join(self.path, "allconstr_{:d}.dat".format(s - 1)))
            actforce = np.loadtxt(os.path.join(self.path, "force.dat"))
            if actforce.shape != actcoor.shape:
                print("WARNING")
            winsize.append(actcoor.shape[0])
            if coor is None:
                coor = actcoor
            else:
                coor = np.append(coor, actcoor, axis=0)
            if force is None:
                force = actforce
            else:
                force = np.append(force, actforce, axis=0)
            sdata = []
            for d in range(1, winsize[-1]+1):
                u = np.loadtxt(os.path.join(self.path, "coll_win_{:d}".format(s),
                                            "data{:d}".format(d)))
                sdata.append(u[self.skip:, :])
            sdata = np.array(sdata)
            if data is None:
                data = sdata
            else:
                data = np.append(data, sdata, axis=0)
        self.data = data
        self.k_val = force
        self.constr_val = coor
        self.winsize = winsize
        return

    def remove_sim(self, index):
        for i in range(len(self.Fprog)):
            self.Fprog[i] = np.delete(self.Fprog[i], index)
        self.UB3d = np.delete(np.delete(self.UB3d, 168, 0), 168, 2)
        self.constr_val = np.delete(self.constr_val, index, 0)
        self.data = np.delete(self.data, index, 0)
        self.denom = np.delete(self.denom, index, 0)
        self.k_val = np.delete(self.k_val, index, 0)
        self.rUepPerSim = np.delete(self.rUepPerSim, index, 0)
        if type(index) is int:
            simsum = 0
            for i in range(len(self.winsize)):
                simsum += self.winsize[i]
                if simsum > index:
                    self.winsize[i] -= 1
        else:
            for j in -np.sort(-index):
                simsum = 0
                for i in range(len(self.winsize)):
                    simsum += self.winsize[i]
                    if simsum > j:
                        self.winsize[i] -= 1
        return

    def plot_data(self):
        colormap = cm.get_cmap('tab20b', self.data.shape[2])
        f, ax = plt.subplots()
        for i in range(self.data.shape[0]):
            xx = np.linspace(i, i + 1, self.data.shape[1])
            for j in range(self.data.shape[2]):
                if i == 0:
                    ax.plot(xx, self.data[i, :, j], color=colormap.colors[j],
                            linewidth=0.5, label="r{:d}".format(j + 1))
                else:
                    ax.plot(xx, self.data[i, :, j], color=colormap.colors[j],
                            linewidth=0.5)
        plt.xlabel("String Points", fontsize=15)
        plt.ylabel("Distance", fontsize=15)
        ax.tick_params(labelsize=12)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   borderaxespad=0.)
        plt.show()
        return

    def calculate_UB(self):
        """
        deprecated
        :return:
        """
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]

        UB = np.empty(shape=(numsims * numsims * datlength), dtype=np.float_)
        kk = 0
        for i in range(numsims):
            for j in range(datlength):
                UB[kk:kk + numsims] = np.exp(
                   -np.sum(0.5 * self.k_val[:, :] * np.square(self.constr_val[:, :] - self.data[i, j, :]),
                           axis=1) / self.KbT)
                kk += numsims
        self.UB = UB
        return

    def calculate_UB3d(self):
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]

        UB = np.empty(shape=(numsims, datlength, numsims), dtype=np.float_)
        for i in range(numsims):
            for j in range(datlength):
                UB[i, j, :] = np.exp(
                   -np.sum(0.5 * self.k_val[:, :] * np.square(self.constr_val[:, :] - self.data[i, j, :]),
                           axis=1) / self.KbT)
        self.UB3d = UB
        return

    def converge(self, threshold=0.01):
        if self.UB is None:
            # self.calculate_UB()
            self.calculate_UB3d()
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]
        if self.Fprog is None:
            Fprog = []
            Fx_old = np.ones(shape=numsims, dtype=np.float_)
        else:
            Fprog = self.Fprog
            Fx_old = Fprog[-1]
        change = 1

        while change > threshold:
            expFx_old = datlength * np.exp(Fx_old / self.KbT)
            # t1 = time()
            # a = self.UB3d * expFx_old
            # a = numba_mult(self.UB3d, expFx_old)
            a = jax_mult(self.UB3d, expFx_old)
            # t2 = time()
            sum = np.sum(a, axis=2)
            # t3 = time()
            denom = np.divide(1, sum, where=sum != 0)
            # Fxf = np.zeros(shape=self.UB3d.shape, dtype=np.float_)
            # t5 = time()
            # Fxf = self.UB3d * denom[:, :, None]
            Fxf = jax_mult_broadcast(self.UB3d, denom[:, :, None])
            # t6 = time()
            Fx = np.sum(Fxf, axis=(0, 1))
            # t7 = time()
            Fx = -self.KbT * np.log(Fx)
            Fx -= Fx[-1]
            Fx_old = Fx
            if len(Fprog) > 1:
                change = np.nanmax(np.abs(Fprog[-1][1:] - Fx[1:]))
            if len(Fprog) > 2:
                prevchange = np.nanmax(np.abs(Fprog[-2][1:] - Fprog[-1][1:]))
                if prevchange < change:
                    print("The iteration started to diverge.")
                    break
            Fprog.append(Fx)
            print(change)
            # print(t2 - t1, t3 - t2, t6 - t5, t7 - t6)
        self.Fprog = Fprog
        return

    def calc_denom(self):
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]
        d = np.zeros(shape=(numsims, datlength))
        for i in range(numsims):
            for j in range(datlength):
                Ubias = np.sum(0.5 * self.k_val[:, :] * np.square(self.constr_val[:, :] - self.data[i, j, :]), axis=1)
                denom = np.sum(datlength * np.exp((self.Fprog[-1] - Ubias) / self.KbT))
                d[i, j] = 1 / denom
        self.denom = d
        return

    def project_2d(self, cv, numbins_q=50):
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]
        q1 = np.sum(self.data * cv[0], axis=2)
        # k_q1 = np.sum(self.constr_val * cv[0], axis=1)
        q2 = np.sum(self.data * cv[1], axis=2)
        # k_q2 = np.sum(self.constr_val * cv[1], axis=1)
        qep = q1 + q2
        qspace12 = create_bins(qep, numbins_q)
        qspace1 = create_bins(q1, numbins_q)
        qspace2 = create_bins(q2, numbins_q)
        Pq12 = np.zeros(shape=numbins_q, dtype=np.float_)
        Pq1 = np.zeros(shape=numbins_q, dtype=np.float_)
        Pq2 = np.zeros(shape=numbins_q, dtype=np.float_)
        Pq2d = np.zeros(shape=(numbins_q, numbins_q), dtype=np.float_)
        PepPersim = np.zeros(shape=(numsims, numbins_q), dtype=np.float_)
        for i in range(numsims):
            for j in range(datlength):
                indq = np.digitize(qep[i, j], qspace12) - 1
                indq1 = np.digitize(q1[i, j], qspace1) - 1
                indq2 = np.digitize(q2[i, j], qspace2) - 1
                Ubias = np.sum(0.5 * self.k_val[:, :] * np.square(self.constr_val[:, :] - self.data[i, j, :]), axis=1)
                denom = np.sum(datlength * np.exp((self.Fprog[-1] - Ubias) / self.KbT))
                Pq12[indq] += 1 / denom
                Pq1[indq1] += 1 / denom
                Pq2[indq2] += 1 / denom
                Pq2d[indq1, indq2] += 1 / denom
                PepPersim[i, indq] += 1 / denom
        rUep = -self.KbT * np.log(Pq12)
        valu = np.min(rUep[:int(numbins_q/2)])
        self.rUep = rUep - valu
        self.rUepPerSim = -self.KbT * np.log(PepPersim) - valu
        self.rUq2d = -self.KbT * np.log(Pq2d) - valu
        self.qspace1 = qspace1
        self.qspace2 = qspace2
        self.qspace12 = qspace12
        return

    def project_1d(self, cv, numbins_q=50):
        numsims = self.data.shape[0]
        qep = np.sum(self.data * cv, axis=2)
        qspace12 = create_bins(qep, numbins_q)
        if self.denom is None:
            self.calc_denom()
        P, PpS = cnt_pop(qep, qspace12, self.denom, numsims=numsims, numbins=numbins_q)
        rUep = -self.KbT * np.log(P)
        valu = np.min(rUep[:int(numbins_q/2)])
        self.rUep = rUep - valu
        self.rUepPerSim = -self.KbT * np.log(PpS) - valu
        self.qspace12 = qspace12
        return -np.max(self.rUep)

    def plot_strings(self, title=None):
        numsims = self.data.shape[0]
        f, a = plt.subplots()
        a.plot(self.qspace12, self.rUep, color="black")
        for i in range(numsims):
            a.plot(self.qspace12, self.rUepPerSim[i], linewidth=0.3)
        if title is None:
            plt.show()
        else:
            plt.title(title)
            plt.savefig("{:s}.png".format(title.replace(" ", "_")))
        return

    def kmeans_colvars(self):
        kmeans = KMeans(n_clusters=24)
        kmeans.fit(self.constr_val)
        return kmeans.cluster_centers_


def bootstrap_error(wham, ratio, cv, threshold=0.0001, iter=100, plotall=False, save=None):
    wham.project_1d(cv)
    full = copy.deepcopy(wham)
    results = []
    for _ in range(iter):
        idx = np.random.randint(full.data.shape[0], size=int((1 - ratio) * full.data.shape[0]))
        wham = copy.deepcopy(full)
        wham.remove_sim(idx)
        wham.converge(threshold=threshold)
        wham.project_1d(cv)
        results.append((wham.qspace12, wham.rUep))
    r = np.array(results).astype(np.float_)
    # r = r[~np.isnan(r).any(axis=(1, 2))]
    # r = r[~np.isinf(r).any(axis=(1, 2))]
    if plotall:
        f, a = plt.subplots()
        for i in range(r.shape[0]):
            a.plot(r[i, 0], r[i, 1])
        plt.show()
    # interpolation
    newU = np.empty(shape=r[:, 1, :].shape, dtype=np.float_)
    for i in range(r.shape[0]):
        newU[i, :] = np.interp(full.qspace12, r[i, 0], r[i, 1])
        # realign
        offset = align(newU[i, :], full.rUep)
        newU[i, :] += offset
    stderr = np.std(newU, axis=0)
    f, a = plt.subplots()
    a.plot(full.qspace12, full.rUep)
    a.fill_between(full.qspace12, full.rUep - stderr, full.rUep + stderr, alpha=0.2)
    # plt.title()
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    return