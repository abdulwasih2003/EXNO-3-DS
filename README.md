## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  ## 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ## 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```python
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![316974447-03f8eed4-910f-4b36-952a-e42f04af61d2](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/d3e75313-da54-4680-8577-bce19611ed72)
## ORDINAL ENCODER
```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![316974700-18d9f738-4373-4195-86c6-1867c0537e42](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/340e80be-983d-4734-a865-71750e298ae4)
```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![316974937-3c39cd75-cbc4-4be0-b6e1-77885f2157aa](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/abf31cec-8da9-47c2-8b21-50c495ebadf2)

## LABEL ENCODER
```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![316975220-d6b15fec-dc0f-4335-aa5e-b7d9c8456359](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/b54f77fd-aa1d-41f8-bec0-b3ae85242cd4)

## ONEHOT ENCODER
```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
```python
df2=pd.concat([df2,enc],axis=1)
df2
```
![316975574-68e576e2-502c-4219-9c2f-82c0cc4be899](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/5f57450a-6e77-4ab9-a09e-e442c161ab48)
```python
pd.get_dummies(df2,columns=["nom_0"])
```
![316975873-f725312f-3f3f-4005-8cc9-64422fd3e140](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/e5a08934-1489-4fba-a221-91e7ee9ccf44)
## BINARY ENCODER
```python
pip install --upgrade category_encoders
```
![316976265-a09f8c01-3e4f-4d5d-93d1-50adcfb34de7](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/ea5469e8-a18e-4d9e-b67d-f37238890d31)
```python
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![316976459-850cfb3a-1268-4876-a209-53f393e11e07](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/ea93d2de-79db-45fb-924f-5f91bc6b32f5)
```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![316976734-48f270d4-9112-4ed1-a819-966fbc9e253c](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/6ec86ecc-fc60-4c35-821f-b79b1753e928)

## TARGET ENCODER
```python
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![316976963-4f88289a-59ac-41b4-a069-362e4ef02f9c](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/4ab61662-972e-47c9-a8f8-32552f81c634)

## DATA TRANSFORMATION
```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/Data_to_Transform.csv')
df
```
![316977331-e534fb21-83c5-4e3d-b308-0e35d7b97b5a](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/9fff71c3-b0ea-48f1-b5f7-18db8f2716f3)
```python
df.skew()
```
![316977432-6fa9e262-5939-4ed8-9431-2f53e5df0249](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/215641d9-6b86-4fe2-ab66-fcec49955cfb)

```python
np.log(df["Highly Positive Skew"])
```
![316977539-ebdff44e-bf3c-4d23-8370-47de6dafc4e1](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/a3a6770c-c9de-419f-8e71-928753fc70b9)

```python
np.reciprocal(df["Moderate Positive Skew"])
```
![316977690-e1545b48-ab16-47da-84bc-f6121b46f43e](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/6d6f12ee-3d44-4eb1-95db-17ab36fe832a)

```python
np.sqrt(df["Highly Positive Skew"])
```
![316977841-1d427996-1049-4286-8703-4352034c1abf](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/81c38e78-d7ff-4e16-91b2-9b635fc6c2ac)
```python
np.square(df["Highly Positive Skew"])
```
![316978073-00c18009-d540-4d86-b440-39377bde361c](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/0dc369d8-481d-4e3c-a9be-075fc66a4235)
```python
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![316978173-f3ef2b2b-9eeb-4062-888b-d118f395436b](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/02f01d84-79c3-408b-898d-6e5182018d52)
```python
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![316978269-ea63ad5a-cb65-4c02-8393-25bfe84571d8](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/d291fcdc-330a-43e1-bba5-c0e33f6c3894)
```python
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![316978503-a704ece5-fa6b-4ee8-8c2f-292747d6bcd8](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/96cabaed-d9f3-4a9b-ac2e-b545d1a6d209)
```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![316978636-a33675eb-6c8f-41fa-b235-c28ee2f6648b](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/80b83139-62f2-45ac-bf5f-cf13abc01143)
```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![316978896-977f0b7b-add6-4cff-a014-d46936459ff9](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/6f8bd450-8309-4444-8ef8-1acec8b726c4)
```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![316980311-381b2529-69d1-449f-9268-acd090addfc1](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/71a84739-3fad-4b6d-b510-c73e8f9c68b1)
```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![316980449-d06eb8f6-3e0e-46e7-b946-262bb767b38d](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/c07ff48f-f4d5-41c4-a486-fff3fc9f16f0)

```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![316980599-a9f1e12e-82e6-4009-ac21-ecce9a0376cc](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/827a31be-f84b-4193-9b7c-941ea2b82777)
```python
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![316980701-fb14ea2a-4cdd-4cc3-bbe4-7c91931be99d](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/61b03f72-3c2b-4b04-914d-c0fc1a21f051)
```python
dt=pd.read_csv("/content/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![316980831-f62a8f38-65c6-468f-820f-9a2d94bc9bec](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/c63211bc-42ed-4621-96e4-4af1dd5a69ea)
```python
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![316980939-671fb915-43b8-4a79-9817-23a1cf973983](https://github.com/KesavDeepak/EXNO-3-DS/assets/139336019/ca2aa355-3200-48c7-b574-0206e8de8c59)

# RESULT:
Thus perform Feature Encoding and Transformation process is executed successfully.
