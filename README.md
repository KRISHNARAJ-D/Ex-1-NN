<H3>NAME:KRISHNARAJ D</H3>
<H3>REGISTER NO:212222230070</H3>
<H3>EX. NO.1</H3>

<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**

Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
### STEP 1:Importing the libraries<BR>
### STEP 2:Importing the dataset<BR>
### STEP 3:Taking care of missing data<BR>
### STEP 4:Encoding categorical data<BR>
### STEP 5:Normalizing the data<BR>
### STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
import pandas as pd                                                 
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         
df.head()
#Find missing values
df.isnull().sum()
df.duplicated().sum()
           
df=df.drop(['Surname', 'Geography','Gender'], axis=1)

scaler=StandardScaler()                                             
df=pd.DataFrame(scaler.fit_transform(df))
df.head()

X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values                     
print("X:",X)
print("Y:",Y)
        
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)
print("Xtrain:" ,Xtrain, "\nXtest:", Xtest)                   
print("Ytrain:" ,Ytrain, "\nYtest:", Ytest)                                  
```
## OUTPUT:
### DATASET
![308667258-5b36fd73-bf5e-44b4-84f6-64fadc748ef2](https://github.com/KRISHNARAJ-D/Ex-1-NN/assets/119559695/9cce3ccb-2f35-4b7e-8104-e02a1d658780)


### NULL VALUES:
![308667302-f9541e19-1be2-4997-814f-7dd190775c6a](https://github.com/KRISHNARAJ-D/Ex-1-NN/assets/119559695/77707c4f-8388-4c38-aaf8-7217d789704c)

### NORMALIZED DATA:

![308667331-0a40e4bc-c9c9-4841-bbed-1f353c39c920](https://github.com/KRISHNARAJ-D/Ex-1-NN/assets/119559695/13f60132-b6d7-4950-8538-0f6f0585ef50)

### DATA SPLITTING:
![308667359-b477384d-6bcd-4f5b-8f58-bc4d04155bbb](https://github.com/KRISHNARAJ-D/Ex-1-NN/assets/119559695/8b53180e-f4ac-498e-beee-18601ea663b6)


### TRAIN AND TEST DATA:
![308667383-6c6a6622-d4b8-4189-aef4-903d7d391ecd](https://github.com/KRISHNARAJ-D/Ex-1-NN/assets/119559695/64a0689a-edc6-4a8d-8183-d34543583d56)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
