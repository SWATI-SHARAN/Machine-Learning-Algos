import numpy as np
import matplotlib.pyplot as plt

mean  = 50
std_devi = 10
sample_size = 1000

normal_data = np.random.normal(loc = mean , scale = std_devi, size = sample_size)
plt.hist(normal_data, bins=30, density=True, edgecolor='green')
plt.title('Normal Distribution Histogram - Generated Data')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()