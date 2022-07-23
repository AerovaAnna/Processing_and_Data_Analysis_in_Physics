import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import astropy.units as u
from astropy.coordinates import SkyCoord



def t(x, tau, mu1, mu2, sigma12, sigma22):
    tau0 = tau
    tau1 = 1 - tau

    T1 = tau0 / np.sqrt(2*np.pi*sigma12) * np.exp(-0.5*((x - mu1)**2)/sigma12)
    T2 = tau1 / np.sqrt(2*np.pi*sigma22) * np.exp(-0.5*((x - mu2)**2)/sigma22)
    T_norm = T1 + T2

    T1 = np.divide(T1, T_norm, out=np.full_like(T_norm, 0.5), where=T_norm!=0)
    T2 = np.divide(T2, T_norm, out=np.full_like(T_norm, 0.5), where=T_norm!=0)

    return np.vstack((T1,T2))

def theta(x, *old):
    T1, T2 = t(x, *old)
    tau = np.sum(T1) / np.sum(T1+T2)
    mu1 = np.sum(x * T1) / np.sum(T1)
    mu2 = np.sum(x * T2) / np.sum(T2)
    sigma12 = np.sum((x - mu1)**2 * T1) / np.sum(T1)
    sigma22 = np.sum((x - mu2)**2 * T2) / np.sum(T2)
    return np.array([tau, mu1, mu2, sigma12, sigma22])

def l(parameters, x):
    tau, mu1, sigma1, mu2, sigma2 = parameters
    T1 = stats.norm.pdf(x, loc=mu1, scale=np.abs(sigma1))
    T2 = stats.norm.pdf(x, loc=mu2, scale=np.abs(sigma2))
    T = tau * T1 + (1 - tau) * T2
    return T, T1, T2

def L(parameters, x):
    T, T1, T2 = l(parameters, x)
    return -np.sum(np.log(np.abs(T)))



def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    T = optimize.minimize(L, np.array([tau, mu1, sigma1, mu2, sigma2]), args=x, tol=rtol)
    return T.x


def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    th = np.array([tau, mu1, mu2, sigma1**2, sigma2**2])
    thold = th
    th = theta(x, *th)
    q = th - thold
    gg = np.linalg.norm(q)
    
    while gg > rtol:
        thold = th
        th = theta(x, *th)
        q = th - thold
        gg = np.linalg.norm(q)
    
    return (th[0], th[1], th[3]**0.5, th[2], th[4]**0.5)




def TT(x, tau1, tau2,  mu1, mu2, sigma12, sigma22):
    x = np.asarray(x)
    T1 = tau1 / (2*np.pi * sigma12) * np.exp(-0.5*np.sum((x-mu1)**2, axis=1) / sigma12)
    T2 = tau2 / (2*np.pi * sigma22) * np.exp(-0.5*np.sum((x-mu2)**2, axis=1) / sigma22)
    T_norm = T1 + T2
    
    T1 = np.divide(T1, T_norm, out=np.full_like(T_norm, 0.5), where=T_norm!=0)
    T2 = np.divide(T2, T_norm, out=np.full_like(T_norm, 0.5), where=T_norm!=0)
    
    return np.vstack((T1,T2))

   


def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2, rtol=1e-5):
    pass


if __name__ == "__main__":
    pass
