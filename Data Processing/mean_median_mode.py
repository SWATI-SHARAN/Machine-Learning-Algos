#Mean, Median, Mode Calculation

#Mean
def calculate_mean(data):
    """Calculate the mean of a list of numbers."""
    if not data:
        return 0
    return sum(data) / len(data)
data_set = [1,2,3,4,5]
mean_value = calculate_mean(data_set)
print(f"Mean: {mean_value}")

#Median
def calculate_median(data):
    """Calculate the median of a list of numbers."""
    data.sort()
    n = len(data)
    mid = n//2
    if n % 2 == 0:
        median = (data[mid - 1] + data[mid]) / 2  
    else:
        median = data[mid]  
    return median
data_set = [1,2,3,4,5]
median_value = calculate_mean(data_set)
print(f"Median: {median_value}")

#Mode
from statistics import mode
def calculate_mode(data):
    """Calculate the mode of a list of numbers."""
data_set = [1,2,3,4,4,4,5,5,5,5,5,5]
mode_value = mode(data_set)
print(f"Mode: {mode_value}")