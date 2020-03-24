import numpy as np
import math
import pandas as pd 

def Bin(N,r,p): 
    return math.factorial(N)/(math.factorial(N-r)*math.factorial(r))*(p**r)*((1-p)**(N-r)) 

 P = np.array([Bin(6,x,0.3) for x in range (7)]) # List with probabilities of x successes

 F = np.cumsum(P) # cumulative distribution of P

 i = np.argsort(P)[::-1] # get the index of the sorted list P in decreassing order

 r = np.array([np.where(i == x)[0][0]+1 for x in range(7)]) # find the possition on the sorted list for each value of r

 F_r = P[np.argsort(r)].cumsum()[r-1]  # cumulative distribution of sorted P

table = pd.DataFrame({'P(r|N=6,p=0.3)':P,'F(r|N=6,p=0.3)':F,'Rank':r,'F(Rank)':F_r}) 

print(table.to_latex())                                                                                                                                                                   