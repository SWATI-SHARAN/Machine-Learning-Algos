import numpy as np
import matplotlib.pyplot as plt
#Histogram Creation
data_distribution = np.random.normal(loc = 50, scale = 10, size = 1000) #loc = 50: Mean of the distribution, scale = 10: Standard deviation,size = 1000: Number of samples 
#NumPy library used to generate random numbers that follow a normal (Gaussian) distribution.
plt.hist(data_distribution, bins = 30, edgecolor = 'black') #data_distribution: Data to be plotted, bins = 30: Divide the range of values into 30 intervals (bars), edgecolor = 'black': Adds black outlines to the bars for better visibility
plt.title('Histogram - Data Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()