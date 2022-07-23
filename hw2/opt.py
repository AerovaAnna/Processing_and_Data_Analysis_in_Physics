from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации
Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""
with open("jla_mub.txt", 'r', encoding='utf-8') as f:
    text = f.read()
words = text.split()
zstr = words[3::2]
mustr = words[4::2]
H_0 = 7.20614866e+03
Om = 5.69102709e-01
z = np.array(zstr, dtype=np.float32)
y = np.array(mustr, dtype=np.float32)

def integral(z, H_0, Om):
    d = []
    for i in range (0,len(z)):
        d.append(((3*10**11/H_0)*(1+z[i]))*integrate.quad(1/np.sqrt((1-Om)*(1+z)**3+Om), 0, z[i],args=Om)[0])
    return np.array(d)

def f(z, H_0, Om):
    f = []
    for i in range (0,len(z)):
        f.append(5*np.log10(integral(z,H_0,Om)[i])-5)
    return np.array(f)

def integralp(z, H_0, Om):
    С = []
    for i in range (0,len(z)):
        С.append(integrate.quad(((1+z)**3)*0.5/(np.sqrt((1-Om)*(1+z)**3+Om))**3, 0, z[i],args=Om)[0])
    return np.array(С)

def J(z, H_0, Om):
    j = np.empty((z.size, 2))
    j[:, 0] = -5/(np.log(10)*H_0)
    j[:, 1] = 5*integral(z, H_0, Om)/(np.log(10)*integralp(z, H_0, Om))
    return j

def gauss_newton(y, f, j, x0, k=1, tol=1e-4):
    i = 1
    x = np.asarray(x0)
    D = y - f(*x)
    cost.append(0.5 * (D @ D))
    j = J(*x)
    delta = np.linalg.inv(j.T @ j) @ j.T @ D
    x = x + k * delta
    cost = [] 
    while np.linalg.norm(delta) <= tol * np.linalg.norm(x):
        i = 1
        x = np.asarray(x0)
         D = y - f(*x)
        cost.append(0.5 * (D @ D))
        j = J(*x)
        delta = np.linalg.inv(j.T @ j) @ j.T @ D
        x = x + k * delta
    cost = np.array(cost)
    pass

def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    pass


if __name__ == "__main__":
    pass

r = gauss(mu, lambda *args: muu(z, *args),lambda *args: J(z, *args),(50, 0.5),k=0.1,tol=1e-4)
print(r.x, r.nfev)


fig = plt.gcf()
plt.grid()
fig.set_size_inches(15,5)
plt.plot(z, f(z, *r.x), label='приближение')
plt.plot(z, y, 'x', label='оригинал')
plt.title('mu(z)')
plt.legend()
fig.savefig('mu-z.png')
