import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

data={
    "Age":[22,25,28,30,35,40,45,29,32,38],
    "Salary":[25000,30000,35000,40000,50000,60000,70000,38000,45000,55000],
    "Department":["IT","HR","Finance","IT","Marketing","HR","Finance","IT","Marketing","HR"],
    "Years_of_Experience":[1,2,4,5,8,12,15,3,6,10]
}
df=pd.DataFrame(data)

print("Original Dataset: ")
print(df)
df.loc[2,"Age"]=np.nan
df.loc[5,"Salary"]=np.nan
df.loc[7,"Department"]=np.nan
df.loc[8,"Years_of_Experience"]=np.nan

print("\nDataset with Missing Values: ")
print(df)
print("\nMissing Values: ")
print(df.isnull().sum())
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Salary"]=df["Salary"].fillna(df["Salary"].mean())
df["Years_of_Experience"]=df["Years_of_Experience"].fillna(df["Years_of_Experience"].mean())

df["Department"]=df["Department"].fillna(df["Department"].mode()[0])
encoder=LabelEncoder()
df["Department"]=encoder.fit_transform(df["Department"])
scaler=StandardScaler()
df[["Age","Salary","Years_of_Experience"]]=scaler.fit_transform(df[["Age","Salary","Years_of_Experience"]])
print("\nPreprocessed Dataset: ")
print(df)
print("\nMissing Values After Preprocessing: ")
print(df.isnull().sum())