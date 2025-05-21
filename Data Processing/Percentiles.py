import numpy as np

def calculate_percentiles(data, percentile):
    return np.percentile(data, percentile)
data_set_1 = [10,15,20,25,30]
percentile_reult_1 = calculate_percentiles(data_set_1, percentile=75)
print("75th Percentile: ", percentile_reult_1 )

data_set_2 = [5,10,15,20,25,30]
percentile_reult_2 = calculate_percentiles(data_set_2, percentile=50)
print("50th Percentile: ", percentile_reult_2 )