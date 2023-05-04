import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('simple_results.csv', names = ['r','ProblemSize','Performance'])
x = df['ProblemSize'].to_numpy() 
y = df['Performance'].to_numpy()
  
# first plot with X and Y data

plt.plot(x, y, label="Simple MatVec")
  
df = pd.read_csv('taco_results.csv', names = ['r','ProblemSize','Performance'])
x1 = df['ProblemSize'].to_numpy() 
y1 = df['Performance'].to_numpy()
  
# second plot with x1 and y1 data
plt.plot(x1, y1, label="TACO MatVec")
plt.legend(loc="upper left")
plt.xlabel("Problem Size")
plt.ylabel("Performance")
plt.title('Performance Plot')
#plt.show()
plt.savefig('TACO_MatVec.png')
