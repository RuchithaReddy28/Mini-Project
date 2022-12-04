# Mini-Project
## Weather-analysis.
## Aim :
Analysis Of Weather In The Data Science.
## Procedure:
Step:1 Importing necessary packages 
Step:2 read the data set
Step:3 Execute the methods 
Step:4 run the program 
Step:5 get the output

## Program And Output:
# Importing necessary packages:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
## read the data set:
```
df=pd.read_csv("weather.csv")
df
```
![202074920-223a336a-361e-440b-b31e-7cca3a1c8a53](https://user-images.githubusercontent.com/93427261/205506122-df8aa8f3-2244-42e9-a895-706bce37c2ff.png)
```
df.head()
```
![202075017-6ab7c3b0-cbe9-4f3e-9f6e-56d3fcdcd8be](https://user-images.githubusercontent.com/93427261/205506138-516ef16d-9130-4142-a593-16ccdb21a90f.png)
```
df.info()
```
![202075224-10709e9c-0f94-4a01-b056-47661fad990b](https://user-images.githubusercontent.com/93427261/205506172-21b77704-dd7e-4697-9ca2-65f916a6e9be.png)
```
df.tail()
```
![202075270-aeb27ea2-cefb-45c4-a909-a60d8a9bb336](https://user-images.githubusercontent.com/93427261/205506184-1372fad4-2142-4675-87e6-38f39d45721c.png)
```
df.describe()
```
![202075365-f97cbbe5-31f6-4cff-9fd9-ebff97e80e50](https://user-images.githubusercontent.com/93427261/205506203-7b6be4e2-b266-47d4-b285-f4a2663eda4c.png)
```
df.shape
```
![202075419-7e9ba26a-104f-45bb-b05e-04ca0eb3bdab](https://user-images.githubusercontent.com/93427261/205506226-6956190f-9496-4d42-874f-8145a928efb0.png)
```
df['weather'].value_counts()
```
![202075488-ee9c5852-bb17-45b0-984a-ecdc225e558d](https://user-images.githubusercontent.com/93427261/205506243-d9d06935-1c20-4548-8d10-927a225da195.png)
# label encoder:
```
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


df['wind'] = le.fit_transform(df['weather'])
df.head(10)
```
![202075544-2ff2b185-77a9-4c4e-a563-9ae28f9cbfc0](https://user-images.githubusercontent.com/93427261/205506305-c2a2c5c2-e610-469c-b773-f8d55191d355.png)
# data cleaning:
```
df.isnull().sum()
```
![202075615-c47bf868-a9e7-4c67-8730-0cabf28a9031](https://user-images.githubusercontent.com/93427261/205506336-05bfdfa5-6bcc-4bf2-a26c-e9501ecad299.png)
```
missing_percentage = (df.isnull().sum())/(df.shape[0])*100
missing_percentage
```
![202075672-562ded93-db62-40c9-82b7-22ffd1cbb391](https://user-images.githubusercontent.com/93427261/205506374-4081a2cb-64a0-4710-b2fe-b06f5bf6dd3e.png)
```
df.duplicated().value_counts()
```
![202075848-abb5493b-bd9f-4f0c-9cc8-9be3065e9787](https://user-images.githubusercontent.com/93427261/205506394-d4a1803a-b075-4fce-afd1-79f4ddccf179.png)
# Univariate Analysis:
```
sns.boxplot(y="wind",data=df)
```
![202076084-b5248e03-6180-4c82-8a94-a33c47486781](https://user-images.githubusercontent.com/93427261/205506431-373c6fa7-334f-4d65-9156-ff152a929686.png)
```
sns.countplot(y="weather",data=df)
```
![202076467-57071e75-3cc2-42f9-8c88-6d3fdcb09e57](https://user-images.githubusercontent.com/93427261/205506455-b2ccc4a7-525f-4716-ad32-1c4cf6b6cf06.png)
```
sns.histplot(y="wind",data=df)
```
![202076570-7d0e696e-32eb-44a4-adc5-f50e0a78beff](https://user-images.githubusercontent.com/93427261/205506468-fb9b61b2-b7b0-42ea-b51f-edf3db4b2d0d.png)
# Multivariate Analysis:
```
sns.scatterplot(df['wind'],df['weather'])
```
![202077081-7637e653-4d25-4849-b2d0-3c2ebcbd90f9](https://user-images.githubusercontent.com/93427261/205506499-9bd80594-04e5-46c0-b6df-cd7c051d5f16.png)
```
sns.barplot(data=df, x='wind', y='precipitation')
```
![202077186-363af693-a67a-490c-8d23-ed695fdf3330](https://user-images.githubusercontent.com/93427261/205506531-50de68b1-9192-4f06-a8a1-a4ccbdcc8635.png)
```
df.corr()
```
![202077255-46a8858c-e616-4b90-84ab-d983e569436f](https://user-images.githubusercontent.com/93427261/205506546-0eef189a-3f3f-4bb2-a6b7-3a9928a6c6d0.png)
```
sns.heatmap(df.corr(),annot=True)
```
![202077324-79db0496-ba14-41c2-afaf-4c1fddaab9de](https://user-images.githubusercontent.com/93427261/205506562-822e799b-87a6-40c7-a2d3-abc6cf2a5ea0.png)
# Data Visualization:
```
plt.figure(figsize=(20, 7))
sns.lineplot(data=df, x='temp_min', y='temp_max')
plt.show()
```
![202077469-d20438b6-2b8e-4ee0-b845-0fee37ea002d](https://user-images.githubusercontent.com/93427261/205506583-61e840be-c06d-4917-9512-b2940e6561ee.png)
```
sns.pointplot(x=df['temp_max'],y=df['temp_min'])
```
![202077572-280474f5-d721-4857-8009-82959be22820](https://user-images.githubusercontent.com/93427261/205506619-fc6cc662-52cf-4b67-974f-920b6a658953.png)
```
sns.kdeplot(x=df['wind'],data=df)
```
![202077649-689749e0-9b44-43de-9f90-48e45724fb2e](https://user-images.githubusercontent.com/93427261/205506654-2a39c6b8-be9e-4d87-a22a-e1d5b14e7065.png)
```
sns.countplot(y="precipitation",data=df)
```
![202077705-a325ad73-9ac3-4ca4-a694-607793f5e798](https://user-images.githubusercontent.com/93427261/205506669-ff4b9d20-470d-4060-8560-a0aece9a491a.png)
# Result:
Hence the program to analyze the data set using data science is applied sucessfully.










