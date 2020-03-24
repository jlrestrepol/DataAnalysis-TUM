import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import pandas as pd
import numpy as np

# Feldman-Cousins

def Poiss_table_FC(N,l,u):
    pmf = np.array([round(poisson.pmf(n,u),4) for n in range(N)])
    u_best = np.array([np.maximum(n,l) for n in range(N)])
    pmf_best = np.array([round(poisson.pmf(n,u_best[n]),4) for n in range(N)])
    r = np.round(pmf / pmf_best,3)
    i = np.argsort(r)[::-1]
    rank = np.array([np.where(i == x)[0][0]+1 for x in range(N)])
    F_r = pmf[np.argsort(rank)].cumsum()[rank-1]
    data = {'P(n|u)':pmf, 'u_best':u_best, 'P(n|u_best)':pmf_best, 'r':r, 'Rank':rank, 'F_r(n|u)':F_r }
    df = pd.DataFrame(data)
    return df    

def FC(df,alpha):
    srtd = df['F_r(n|u)'].reindex(np.argsort(df['Rank']))
    upper = np.min(srtd[srtd>=alpha])
    array = np.argwhere(srtd == upper)
    sml = srtd[:array[0][0]+1]
    return sml.index.to_numpy()

intervals_FC = []
for uu in np.arange(3.1,15, 0.1):
    df_FC = Poiss_table_FC(N=24, l=3.0, u=uu)
    FC_int = FC(df_FC,0.90)
    intervals_FC.append((np.min(FC_int),np.max(FC_int)))
intervals_FC = np.array(intervals_FC)

#Smallest Interval

def Poiss_table(N,u):
    pmf = np.array([round(poisson.pmf(n,u),4) for n in range(N)])
    cdf = np.array([round(poisson.cdf(n,u),4) for n in range(N)])
    i = np.argsort(pmf)[::-1]
    r = np.array([np.where(i == x)[0][0]+1 for x in range(N)])
    F_r = pmf[np.argsort(r)].cumsum()[r-1]
    data = {'P(n|u)':pmf, 'F(n|u)':cdf, 'R':r, 'F_r(n|u)':F_r }
    df = pd.DataFrame(data)
    return df    

def smallest(df,alpha):
    srtd = df['F_r(n|u)'].reindex(np.argsort(df['R']))
    upper = np.min(srtd[srtd>=alpha])
    array = np.argwhere(srtd == upper)
    sml = srtd[:array[0][0]+1]
    return sml.index.to_numpy()

intervals_sm = []
for uu in np.arange(3.1,15, 0.1):
    df = Poiss_table(N=24,u=uu)
    sml = smallest(df,0.90)
    intervals_sm.append((np.min(sml),np.max(sml)))
intervals_sm = np.array(intervals_sm)

#Credibility interval
def Post(nu,n,l):
    num = np.math.exp(-nu)*(l+nu)**n
    den = np.math.factorial(n)*np.sum([l**i/np.math.factorial(i) for i in range(n+1)])
    return num/den


def F_post(nu,n,l):
    num = np.math.exp(-nu)*np.sum([(l+nu)**i/np.math.factorial(i) for i in range(n+1)])
    den = np.sum([l**i/np.math.factorial(i) for i in range(n+1)])
    return 1-num/den

def my_fun(z):
    nu1 = z[0]
    nu2 = z[1]

    f = np.zeros(2)
    f[0] = F_post(nu2,n=nn,l=3) - F_post(nu1,n=nn,l=3) - 0.9
    f[1] = Post(nu2,n=nn,l=3) - Post(nu1,n=nn,l=3)
    return np.dot(f,f)

def my_cons(z):
    nu1 = z[0]
    nu2 = z[1]
    f = np.zeros(2)
    f[0] = nu1
    f[1] = nu2
    return f

from scipy.optimize import minimize


cons = {'type' : 'ineq', 'fun': my_cons}
intervals = []
for nn in range(11):
    intervals.append(minimize(my_fun, (1, 0), method='SLSQP', constraints=cons, tol=1e-10).x)
intervals = np.array(intervals)

# Integrated plot

plt.figure(figsize=[8.0,6.0])
plt.xticks(ticks = np.arange(len(df_FC)*10+1, step = 1), labels=np.arange(int(len(df_FC)*10+1)))
plt.yticks(ticks = np.arange(len(intervals_FC)+1, step = 10), labels=np.arange(int(len(intervals_FC)/10+1)))
plt.ylabel('\u03BD')
plt.xlabel('n')
plt.grid()
plt.title(r'Poisson 90% CL Bands for Feldam-Cousins Interval, Bayesian and Smallest Interval with $\lambda$ = 3')

plt.hlines(y = np.arange(len(intervals_FC)-1), xmin = intervals_FC[:,0], xmax = intervals_FC[1:,0], 
           colors='red')
plt.hlines(y = np.arange(len(intervals_FC)-1), xmin = intervals_FC[:,1], xmax = intervals_FC[1:,1], 
           colors='blue')

plt.plot(np.arange(len(intervals)),intervals[:,0]*10,'go')
plt.plot(np.arange(len(intervals)),intervals[:,1]*10,'go')

a=0
for e in intervals_sm:
    plt.fill_betweenx(y=np.array([a,a+1]), x1 = np.array([e[0],e[0]]), x2 = np.array(e[1],e[1]), color = 'gray')
    a+=1