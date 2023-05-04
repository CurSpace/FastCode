import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('nn_without_taco.csv', names = ['r','ProblemSize','Performance'])
x = df['ProblemSize'].to_numpy() 
y = df['Performance'].to_numpy()
  
# first plot with X and Y data

plt.plot(x, y, label="Simple NN")
  
df = pd.read_csv('nn_with_taco.csv', names = ['r','ProblemSize','Performance'])
x1 = df['ProblemSize'].to_numpy() 
y1 = df['Performance'].to_numpy()
  
# second plot with x1 and y1 data
plt.plot(x1, y1, label="TACO NN")
plt.legend(loc="upper left")
plt.xlabel("Problem Size")
plt.ylabel("Performance")
plt.title('Performance Plot')
#plt.show()
plt.savefig('TACONN.png')
