#original code from github: https://github.com/rostaresearch/enhanced-sampling-workshop-2022/blob/main/Day1/src/dham.py
#modified by TW on 28th July 2023
#note that in this code we presume the bias is 10 gaussian functions added together.
#returns the Markov Matrix, free energy surface probed by DHAM. 


#note this is now in 2D.

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.linalg import eig
from scipy.optimize import minimize


def gaussian_2d(x, y, ax, bx, by, cx, cy):
    return np.exp(-((x-bx)**2/(2*cx**2) + (y-by)**2/(2*cy**2)))

def rmsd(offset, a, b):
    return np.sqrt(np.mean(np.square((a + offset) - b)))


def align(query, ref):
    offset = -10.0
    res = minimize(rmsd, offset, args=(query, ref))
    print(res.x[0], res.fun)
    return res.x[0]


def count_transitions(b, numbins, lagtime, endpt=None):
    if endpt is None:
        endpt = b
    Ntr = np.zeros(shape=(b.shape[0], numbins, numbins), dtype=np.int64)  # number of transitions
    for k in range(b.shape[0]):
        for i in range(lagtime, b.shape[1]):
            try:
                Ntr[k, b[k, i - lagtime], endpt[k, i]] += 1
            except IndexError:
                continue
    sumtr = np.sum(Ntr, axis=0)
    trvec = np.sum(Ntr, axis=2)
    sumtr = 0.5 * (sumtr + np.transpose(sumtr)) #disable for original DHAM, enable for DHAM_sym
    # anti = 0.5 * (sumtr - np.transpose(sumtr))
    # print("Degree of symmetry:",
    #       (np.linalg.norm(sym) - np.linalg.norm(anti)) / (np.linalg.norm(sym) + np.linalg.norm(anti)))
    #plt.contourf(sumtr.real)
    #plt.colorbar()
    return sumtr.real, trvec


class DHAM:
    KbT = 0.001987204259 * 300  # energy unit: kcal/mol
    epsilon = 0.00001
    data = None
    vel = None
    datlength = None
    k_val = None
    constr_val = None
    qspace = None
    numbins = 20
    lagtime = 1

    def __init__(self, gaussian_params):
        #unpack it to self.a, self.bx, self.by, self.cx, self.cy
        num_gaussian = len(gaussian_params)//5
        self.a = gaussian_params[:num_gaussian]
        self.bx = gaussian_params[num_gaussian:2*num_gaussian]
        self.by = gaussian_params[2*num_gaussian:3*num_gaussian]
        self.cx = gaussian_params[3*num_gaussian:4*num_gaussian]
        self.cy = gaussian_params[4*num_gaussian:5*num_gaussian]
        x,y = np.meshgrid(np.linspace(-3,3, self.numbins), np.linspace(-3,3, self.numbins))
        self.x = x
        self.y = y
        self.N = 20
        return

    def setup(self, CV, T, prop_index, time_tag):
        self.data = CV
        self.KbT = 0.001987204259 * T
        self.prop_index = prop_index
        self.time_tag = time_tag
        return

    def build_MM(self, sumtr, trvec, biased=False):
        N = self.numbins
        MM = np.empty(shape=(N*N, N*N), dtype=np.longdouble)
        X,Y = np.meshgrid(np.linspace(-3,3, self.numbins), np.linspace(-3,3, self.numbins), indexing='ij')
        if biased:
            MM = np.zeros(shape=(N*N, N*N), dtype=np.longdouble)
            #compute the total bias u.
            u = np.zeros_like(self.x)
            for n in range(len(self.a)):
                u += gaussian_2d(self.x, self.y, self.a[n], self.bx[n], self.by[n], self.cx[n], self.cy[n])
            
            for i in range(N*N):
                for j in range(N*N):
                    if sumtr[i, j] > 0:
                        sump1 = 0.0
                        i_x, i_y = np.unravel_index(i, (20, 20), order='C')
                        j_x, j_y = np.unravel_index(j, (20, 20), order='C')

                        for k in range(trvec.shape[0]):
                            if trvec[k, i] > 0:
                                sump1 += trvec[k, i] * np.exp(-(u[j_x, j_y] - u[i_x, i_y]) / (2*self.KbT))
                        if sump1 > 0:
                            MM[i, j] = sumtr[i, j] / sump1
                        else:
                            MM[i, j] = 0

            #epsilon_offset = 1e-15
            MM = MM / (np.sum(MM, axis=1)[:, None] + 1e-15) #normalize the M matrix #this is returning NaN?.
            """for i in range(MM.shape[0]):
                if np.sum(MM[i, :]) > 0:
                    MM[i, :] = MM[i, :] / np.sum(MM[i, :])
                else:
                    MM[i, :] = 0"""
        else:
            raise NotImplementedError
        
        #plt.contourf(MM.real)
        return MM

    def run(self, plot=True, adjust=True, biased=False, conversion=2E-13):
        """

        :param plot:
        :param adjust:
        :param biased:
        :param conversion: from timestep to seconds
        :return:
        """
        v_min = np.nanmin(self.data) - self.epsilon
        v_max = np.nanmax(self.data) + self.epsilon
        
        #digitialize the data into 2D mesh.
        b = np.digitize(self.data, np.linspace(0, (self.N**2)+1, self.numbins*self.numbins+1))
        b = b.reshape(1,-1)
        #here we check the b trajectory.
        #unravel it.
        #b_test = b.reshape(-1)
        #b_test = np.unravel_index(b, (self.numbins, self.numbins), order='C')
        #plt.plot(b_test[0], b_test[1], 'o')
        #plt.xlim(0, 13)
        #plt.ylim(0, 13)
        

        sumtr, trvec = count_transitions(b, self.numbins * self.numbins, self.lagtime)

        MM = self.build_MM(sumtr, trvec, biased)
        #MM = MM.T  # to adapt our script pattern.
        d, v = eig(MM.T)
        mpeq = v[:, np.where(d == np.max(d))[0][0]]
        mpeq = mpeq / np.sum(mpeq)
        mpeq = mpeq.real
        #rate = np.float_(- self.lagtime * conversion / np.log(d[np.argsort(d)[-2]]))
        mU2 = - self.KbT * np.log(mpeq)
        #dG = np.max(mU2[:int(self.numbins)])
        #A = rate / np.exp(- dG / self.KbT)

        #from util_2d import compute_free_energy
        #mU2 = compute_free_energy(MM, self.KbT)[1]

        plt.figure()
        plt.imshow(mU2.reshape(self.N, self.N), cmap="coolwarm", extent=[-3,3,-3,3])
        plt.colorbar()
        plt.savefig(f"./figs/{self.time_tag}_{self.prop_index}_DHAM.png")
        #plt.show()
        plt.close()
        
        
        """
        from util_2d import compute_free_energy, Markov_mfpt_calc, kemeny_constant_check
        peq_M, F, evectors, evalues, evalues_sorted, index = compute_free_energy(MM, self.KbT)
        #mfpts = Markov_mfpt_calc(peq_M, MM)
        #kemeny_constant_check(mfpts, peq_M)
        if plot:
            F_2D = F.reshape(self.N, self.N).real
            F_2D_masked = np.ma.masked_invalid(F_2D)
            plt.contourf(self.x, self.y, F_2D_masked.T) #, label="reconstructed M by DHAMsym")
            plt.colorbar()
            plt.show()
            #plt.title("Lagtime={0:d} Nbins={1:d}".format(self.lagtime, self.numbins))
            #plt.show()
        """
        return mU2, MM