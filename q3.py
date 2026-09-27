import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = np.array([
    [22, 25000],
    [25, 30000],
    [28, 35000],
    [30, 40000],
    [35, 50000],
    [40, 60000],
    [45, 70000]
])

print("Original data: ")
print(data)

standard_scaler=StandardScaler()
standard_data=standard_scaler.fit_transform(data)
print("\nData after StandarScaler: ")
print(standard_data)
print("\nStandardScaler Range: ")
print("Minimum: ",standard_data.min(axis=0))
print("Maximum: ",standard_data.max(axis=0))

minmax_scaler=MinMaxScaler()
minmax_data=minmax_scaler.fit_transform(data)
print("\nData after MinMaxScaler: ")
print(minmax_data)
print("\nMinMaxScaler Range: ")
print("Minimum: ",minmax_data.min(axis=0))
print("Maximum: ",minmax_data.max(axis=0))
