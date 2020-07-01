## The Role of Analytics in Human Resources

#### Objectives
* Uncorver the factors that contribute to employer attrition rate
* Provide reccommendations on areas to invest to decrease the rate of attrition
* Create a Model to predict the probality of employer leaving the company


#### Data Source
* [Data Source](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

#### About the data

This is a fictional data set created by IBM data scientists.

* Education : 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'

* EnvironmentSatisfaction : 1 'Low' 2 'Medium' 3 'High' 4 'Very High'

* JobInvolvement :  1 'Low' 2 'Medium' 3 'High' 4 'Very High'

* JobSatisfaction :  1 'Low' 2 'Medium' 3 'High' 4 'Very High'

* PerformanceRating :  1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'

* RelationshipSatisfaction : 1 'Low' 2 'Medium' 3 'High' 4 'Very High'

* WorkLifeBalance : 1 'Bad' 2 'Good' 3 'Better' 4 'Best'

***
#### Import Packages
***


```python
import pandas as pd # for tabular data manipulation
import numpy as np # for numeric computing
import matplotlib.pyplot as plt # for ploting /visualization
import matplotlib as matplot # visulization
import seaborn as sns # visualization

%matplotlib inline
```

***
#### Read the data
***


```python
hrdata = pd.read_csv("./data/HR-Employee-Attrition.csv")
hrdata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



***
### Data Quality Check
***


```python
## check for missing values from the datasets
hrdata.isnull().any()
```




    Age                         False
    Attrition                   False
    BusinessTravel              False
    DailyRate                   False
    Department                  False
    DistanceFromHome            False
    Education                   False
    EducationField              False
    EmployeeCount               False
    EmployeeNumber              False
    EnvironmentSatisfaction     False
    Gender                      False
    HourlyRate                  False
    JobInvolvement              False
    JobLevel                    False
    JobRole                     False
    JobSatisfaction             False
    MaritalStatus               False
    MonthlyIncome               False
    MonthlyRate                 False
    NumCompaniesWorked          False
    Over18                      False
    OverTime                    False
    PercentSalaryHike           False
    PerformanceRating           False
    RelationshipSatisfaction    False
    StandardHours               False
    StockOptionLevel            False
    TotalWorkingYears           False
    TrainingTimesLastYear       False
    WorkLifeBalance             False
    YearsAtCompany              False
    YearsInCurrentRole          False
    YearsSinceLastPromotion     False
    YearsWithCurrManager        False
    dtype: bool



***
**There is no missing values from the dataset. This is because the datasets is a fictional dataset created by the IBM data scientist.**
***


```python
## check for the data detail information
hrdata.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 35 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   Age                       1470 non-null   int64 
     1   Attrition                 1470 non-null   object
     2   BusinessTravel            1470 non-null   object
     3   DailyRate                 1470 non-null   int64 
     4   Department                1470 non-null   object
     5   DistanceFromHome          1470 non-null   int64 
     6   Education                 1470 non-null   int64 
     7   EducationField            1470 non-null   object
     8   EmployeeCount             1470 non-null   int64 
     9   EmployeeNumber            1470 non-null   int64 
     10  EnvironmentSatisfaction   1470 non-null   int64 
     11  Gender                    1470 non-null   object
     12  HourlyRate                1470 non-null   int64 
     13  JobInvolvement            1470 non-null   int64 
     14  JobLevel                  1470 non-null   int64 
     15  JobRole                   1470 non-null   object
     16  JobSatisfaction           1470 non-null   int64 
     17  MaritalStatus             1470 non-null   object
     18  MonthlyIncome             1470 non-null   int64 
     19  MonthlyRate               1470 non-null   int64 
     20  NumCompaniesWorked        1470 non-null   int64 
     21  Over18                    1470 non-null   object
     22  OverTime                  1470 non-null   object
     23  PercentSalaryHike         1470 non-null   int64 
     24  PerformanceRating         1470 non-null   int64 
     25  RelationshipSatisfaction  1470 non-null   int64 
     26  StandardHours             1470 non-null   int64 
     27  StockOptionLevel          1470 non-null   int64 
     28  TotalWorkingYears         1470 non-null   int64 
     29  TrainingTimesLastYear     1470 non-null   int64 
     30  WorkLifeBalance           1470 non-null   int64 
     31  YearsAtCompany            1470 non-null   int64 
     32  YearsInCurrentRole        1470 non-null   int64 
     33  YearsSinceLastPromotion   1470 non-null   int64 
     34  YearsWithCurrManager      1470 non-null   int64 
    dtypes: int64(26), object(9)
    memory usage: 402.1+ KB
    


```python
## Check for data type inconsistencies
hrdata.dtypes
```




    Age                          int64
    Attrition                   object
    BusinessTravel              object
    DailyRate                    int64
    Department                  object
    DistanceFromHome             int64
    Education                    int64
    EducationField              object
    EmployeeCount                int64
    EmployeeNumber               int64
    EnvironmentSatisfaction      int64
    Gender                      object
    HourlyRate                   int64
    JobInvolvement               int64
    JobLevel                     int64
    JobRole                     object
    JobSatisfaction              int64
    MaritalStatus               object
    MonthlyIncome                int64
    MonthlyRate                  int64
    NumCompaniesWorked           int64
    Over18                      object
    OverTime                    object
    PercentSalaryHike            int64
    PerformanceRating            int64
    RelationshipSatisfaction     int64
    StandardHours                int64
    StockOptionLevel             int64
    TotalWorkingYears            int64
    TrainingTimesLastYear        int64
    WorkLifeBalance              int64
    YearsAtCompany               int64
    YearsInCurrentRole           int64
    YearsSinceLastPromotion      int64
    YearsWithCurrManager         int64
    dtype: object



***
**There is consistency in the data type**


There are 26 numeric columns and 9 discrete columns
***

***
### Exploratory Data Analysis
***


```python
## how many employer in the datasets
hrdata.shape
```




    (1470, 35)




```python
## calculate the attrition rate of the company.
attrition_rate = hrdata.Attrition.value_counts() / hrdata.shape[0]
attrition_rate.plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1381b843b48>




![png](./templates/output_17_1.png)


**This is an imbalance datasets. 16% of the employer is classifield has YES in attrition rate while 83% is classifield as NO in the attrition rate.**

We are are dealing with class inbalance


```python
## Summary statistics of the employer
hrdata.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1470.0</td>
      <td>36.923810</td>
      <td>9.135373</td>
      <td>18.0</td>
      <td>30.00</td>
      <td>36.0</td>
      <td>43.00</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>DailyRate</th>
      <td>1470.0</td>
      <td>802.485714</td>
      <td>403.509100</td>
      <td>102.0</td>
      <td>465.00</td>
      <td>802.0</td>
      <td>1157.00</td>
      <td>1499.0</td>
    </tr>
    <tr>
      <th>DistanceFromHome</th>
      <td>1470.0</td>
      <td>9.192517</td>
      <td>8.106864</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>7.0</td>
      <td>14.00</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>1470.0</td>
      <td>2.912925</td>
      <td>1.024165</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>4.00</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>EmployeeCount</th>
      <td>1470.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>EmployeeNumber</th>
      <td>1470.0</td>
      <td>1024.865306</td>
      <td>602.024335</td>
      <td>1.0</td>
      <td>491.25</td>
      <td>1020.5</td>
      <td>1555.75</td>
      <td>2068.0</td>
    </tr>
    <tr>
      <th>EnvironmentSatisfaction</th>
      <td>1470.0</td>
      <td>2.721769</td>
      <td>1.093082</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>4.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>HourlyRate</th>
      <td>1470.0</td>
      <td>65.891156</td>
      <td>20.329428</td>
      <td>30.0</td>
      <td>48.00</td>
      <td>66.0</td>
      <td>83.75</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>JobInvolvement</th>
      <td>1470.0</td>
      <td>2.729932</td>
      <td>0.711561</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>JobLevel</th>
      <td>1470.0</td>
      <td>2.063946</td>
      <td>1.106940</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>3.00</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>JobSatisfaction</th>
      <td>1470.0</td>
      <td>2.728571</td>
      <td>1.102846</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>4.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>1470.0</td>
      <td>6502.931293</td>
      <td>4707.956783</td>
      <td>1009.0</td>
      <td>2911.00</td>
      <td>4919.0</td>
      <td>8379.00</td>
      <td>19999.0</td>
    </tr>
    <tr>
      <th>MonthlyRate</th>
      <td>1470.0</td>
      <td>14313.103401</td>
      <td>7117.786044</td>
      <td>2094.0</td>
      <td>8047.00</td>
      <td>14235.5</td>
      <td>20461.50</td>
      <td>26999.0</td>
    </tr>
    <tr>
      <th>NumCompaniesWorked</th>
      <td>1470.0</td>
      <td>2.693197</td>
      <td>2.498009</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>4.00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>PercentSalaryHike</th>
      <td>1470.0</td>
      <td>15.209524</td>
      <td>3.659938</td>
      <td>11.0</td>
      <td>12.00</td>
      <td>14.0</td>
      <td>18.00</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>PerformanceRating</th>
      <td>1470.0</td>
      <td>3.153741</td>
      <td>0.360824</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>RelationshipSatisfaction</th>
      <td>1470.0</td>
      <td>2.712245</td>
      <td>1.081209</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>4.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>StandardHours</th>
      <td>1470.0</td>
      <td>80.000000</td>
      <td>0.000000</td>
      <td>80.0</td>
      <td>80.00</td>
      <td>80.0</td>
      <td>80.00</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>StockOptionLevel</th>
      <td>1470.0</td>
      <td>0.793878</td>
      <td>0.852077</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>TotalWorkingYears</th>
      <td>1470.0</td>
      <td>11.279592</td>
      <td>7.780782</td>
      <td>0.0</td>
      <td>6.00</td>
      <td>10.0</td>
      <td>15.00</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>TrainingTimesLastYear</th>
      <td>1470.0</td>
      <td>2.799320</td>
      <td>1.289271</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>WorkLifeBalance</th>
      <td>1470.0</td>
      <td>2.761224</td>
      <td>0.706476</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>YearsAtCompany</th>
      <td>1470.0</td>
      <td>7.008163</td>
      <td>6.126525</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>5.0</td>
      <td>9.00</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>YearsInCurrentRole</th>
      <td>1470.0</td>
      <td>4.229252</td>
      <td>3.623137</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>7.00</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>YearsSinceLastPromotion</th>
      <td>1470.0</td>
      <td>2.187755</td>
      <td>3.222430</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>3.00</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>YearsWithCurrManager</th>
      <td>1470.0</td>
      <td>4.123129</td>
      <td>3.568136</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>7.00</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## diplay the mean summary of the attrition rate of the company, to observed the group difference
hrdata_summary = hrdata.groupby('Attrition')
hrdata_summary.mean().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Attrition</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>37.561233</td>
      <td>33.607595</td>
    </tr>
    <tr>
      <th>DailyRate</th>
      <td>812.504461</td>
      <td>750.362869</td>
    </tr>
    <tr>
      <th>DistanceFromHome</th>
      <td>8.915653</td>
      <td>10.632911</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>2.927007</td>
      <td>2.839662</td>
    </tr>
    <tr>
      <th>EmployeeCount</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>EmployeeNumber</th>
      <td>1027.656123</td>
      <td>1010.345992</td>
    </tr>
    <tr>
      <th>EnvironmentSatisfaction</th>
      <td>2.771290</td>
      <td>2.464135</td>
    </tr>
    <tr>
      <th>HourlyRate</th>
      <td>65.952149</td>
      <td>65.573840</td>
    </tr>
    <tr>
      <th>JobInvolvement</th>
      <td>2.770479</td>
      <td>2.518987</td>
    </tr>
    <tr>
      <th>JobLevel</th>
      <td>2.145985</td>
      <td>1.637131</td>
    </tr>
    <tr>
      <th>JobSatisfaction</th>
      <td>2.778589</td>
      <td>2.468354</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>6832.739659</td>
      <td>4787.092827</td>
    </tr>
    <tr>
      <th>MonthlyRate</th>
      <td>14265.779400</td>
      <td>14559.308017</td>
    </tr>
    <tr>
      <th>NumCompaniesWorked</th>
      <td>2.645580</td>
      <td>2.940928</td>
    </tr>
    <tr>
      <th>PercentSalaryHike</th>
      <td>15.231144</td>
      <td>15.097046</td>
    </tr>
    <tr>
      <th>PerformanceRating</th>
      <td>3.153285</td>
      <td>3.156118</td>
    </tr>
    <tr>
      <th>RelationshipSatisfaction</th>
      <td>2.733982</td>
      <td>2.599156</td>
    </tr>
    <tr>
      <th>StandardHours</th>
      <td>80.000000</td>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>StockOptionLevel</th>
      <td>0.845093</td>
      <td>0.527426</td>
    </tr>
    <tr>
      <th>TotalWorkingYears</th>
      <td>11.862936</td>
      <td>8.244726</td>
    </tr>
    <tr>
      <th>TrainingTimesLastYear</th>
      <td>2.832928</td>
      <td>2.624473</td>
    </tr>
    <tr>
      <th>WorkLifeBalance</th>
      <td>2.781022</td>
      <td>2.658228</td>
    </tr>
    <tr>
      <th>YearsAtCompany</th>
      <td>7.369019</td>
      <td>5.130802</td>
    </tr>
    <tr>
      <th>YearsInCurrentRole</th>
      <td>4.484185</td>
      <td>2.902954</td>
    </tr>
    <tr>
      <th>YearsSinceLastPromotion</th>
      <td>2.234388</td>
      <td>1.945148</td>
    </tr>
    <tr>
      <th>YearsWithCurrManager</th>
      <td>4.367397</td>
      <td>2.852321</td>
    </tr>
  </tbody>
</table>
</div>




```python
## check the summary std
hrdata_summary.std().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Attrition</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>8.888360</td>
      <td>9.689350</td>
    </tr>
    <tr>
      <th>DailyRate</th>
      <td>403.208379</td>
      <td>401.899519</td>
    </tr>
    <tr>
      <th>DistanceFromHome</th>
      <td>8.012633</td>
      <td>8.452525</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>1.027002</td>
      <td>1.008244</td>
    </tr>
    <tr>
      <th>EmployeeCount</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>EmployeeNumber</th>
      <td>606.217074</td>
      <td>580.750572</td>
    </tr>
    <tr>
      <th>EnvironmentSatisfaction</th>
      <td>1.071132</td>
      <td>1.169791</td>
    </tr>
    <tr>
      <th>HourlyRate</th>
      <td>20.380754</td>
      <td>20.099958</td>
    </tr>
    <tr>
      <th>JobInvolvement</th>
      <td>0.692050</td>
      <td>0.773405</td>
    </tr>
    <tr>
      <th>JobLevel</th>
      <td>1.117933</td>
      <td>0.940594</td>
    </tr>
    <tr>
      <th>JobSatisfaction</th>
      <td>1.093277</td>
      <td>1.118058</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>4818.208001</td>
      <td>3640.210367</td>
    </tr>
    <tr>
      <th>MonthlyRate</th>
      <td>7102.260749</td>
      <td>7208.153264</td>
    </tr>
    <tr>
      <th>NumCompaniesWorked</th>
      <td>2.460090</td>
      <td>2.678519</td>
    </tr>
    <tr>
      <th>PercentSalaryHike</th>
      <td>3.639511</td>
      <td>3.770294</td>
    </tr>
    <tr>
      <th>PerformanceRating</th>
      <td>0.360408</td>
      <td>0.363735</td>
    </tr>
    <tr>
      <th>RelationshipSatisfaction</th>
      <td>1.071603</td>
      <td>1.125437</td>
    </tr>
    <tr>
      <th>StandardHours</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>StockOptionLevel</th>
      <td>0.841985</td>
      <td>0.856361</td>
    </tr>
    <tr>
      <th>TotalWorkingYears</th>
      <td>7.760719</td>
      <td>7.169204</td>
    </tr>
    <tr>
      <th>TrainingTimesLastYear</th>
      <td>1.293585</td>
      <td>1.254784</td>
    </tr>
    <tr>
      <th>WorkLifeBalance</th>
      <td>0.681907</td>
      <td>0.816453</td>
    </tr>
    <tr>
      <th>YearsAtCompany</th>
      <td>6.096298</td>
      <td>5.949984</td>
    </tr>
    <tr>
      <th>YearsInCurrentRole</th>
      <td>3.649402</td>
      <td>3.174827</td>
    </tr>
    <tr>
      <th>YearsSinceLastPromotion</th>
      <td>3.234762</td>
      <td>3.153077</td>
    </tr>
    <tr>
      <th>YearsWithCurrManager</th>
      <td>3.594116</td>
      <td>3.143349</td>
    </tr>
  </tbody>
</table>
</div>




```python
## check the columns list
hrdata.columns
```




    Index(['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
           'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
           'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
           'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
           'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
           'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
           'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
           'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
           'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
           'YearsWithCurrManager'],
          dtype='object')



***
#### Correlation Matrix
***


```python
## create a correlation matrix

plt.figure(figsize=(10,8))

cor = hrdata.corr()

sns.heatmap(cor, xticklabels=cor.columns.values, yticklabels= cor.columns.values)
plt.title("Heatmap of Correlation Matrix")
cor
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>0.010661</td>
      <td>-0.001686</td>
      <td>0.208034</td>
      <td>NaN</td>
      <td>-0.010145</td>
      <td>0.010146</td>
      <td>0.024287</td>
      <td>0.029820</td>
      <td>0.509604</td>
      <td>...</td>
      <td>0.053535</td>
      <td>NaN</td>
      <td>0.037510</td>
      <td>0.680381</td>
      <td>-0.019621</td>
      <td>-0.021490</td>
      <td>0.311309</td>
      <td>0.212901</td>
      <td>0.216513</td>
      <td>0.202089</td>
    </tr>
    <tr>
      <th>DailyRate</th>
      <td>0.010661</td>
      <td>1.000000</td>
      <td>-0.004985</td>
      <td>-0.016806</td>
      <td>NaN</td>
      <td>-0.050990</td>
      <td>0.018355</td>
      <td>0.023381</td>
      <td>0.046135</td>
      <td>0.002966</td>
      <td>...</td>
      <td>0.007846</td>
      <td>NaN</td>
      <td>0.042143</td>
      <td>0.014515</td>
      <td>0.002453</td>
      <td>-0.037848</td>
      <td>-0.034055</td>
      <td>0.009932</td>
      <td>-0.033229</td>
      <td>-0.026363</td>
    </tr>
    <tr>
      <th>DistanceFromHome</th>
      <td>-0.001686</td>
      <td>-0.004985</td>
      <td>1.000000</td>
      <td>0.021042</td>
      <td>NaN</td>
      <td>0.032916</td>
      <td>-0.016075</td>
      <td>0.031131</td>
      <td>0.008783</td>
      <td>0.005303</td>
      <td>...</td>
      <td>0.006557</td>
      <td>NaN</td>
      <td>0.044872</td>
      <td>0.004628</td>
      <td>-0.036942</td>
      <td>-0.026556</td>
      <td>0.009508</td>
      <td>0.018845</td>
      <td>0.010029</td>
      <td>0.014406</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>0.208034</td>
      <td>-0.016806</td>
      <td>0.021042</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.042070</td>
      <td>-0.027128</td>
      <td>0.016775</td>
      <td>0.042438</td>
      <td>0.101589</td>
      <td>...</td>
      <td>-0.009118</td>
      <td>NaN</td>
      <td>0.018422</td>
      <td>0.148280</td>
      <td>-0.025100</td>
      <td>0.009819</td>
      <td>0.069114</td>
      <td>0.060236</td>
      <td>0.054254</td>
      <td>0.069065</td>
    </tr>
    <tr>
      <th>EmployeeCount</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>EmployeeNumber</th>
      <td>-0.010145</td>
      <td>-0.050990</td>
      <td>0.032916</td>
      <td>0.042070</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.017621</td>
      <td>0.035179</td>
      <td>-0.006888</td>
      <td>-0.018519</td>
      <td>...</td>
      <td>-0.069861</td>
      <td>NaN</td>
      <td>0.062227</td>
      <td>-0.014365</td>
      <td>0.023603</td>
      <td>0.010309</td>
      <td>-0.011240</td>
      <td>-0.008416</td>
      <td>-0.009019</td>
      <td>-0.009197</td>
    </tr>
    <tr>
      <th>EnvironmentSatisfaction</th>
      <td>0.010146</td>
      <td>0.018355</td>
      <td>-0.016075</td>
      <td>-0.027128</td>
      <td>NaN</td>
      <td>0.017621</td>
      <td>1.000000</td>
      <td>-0.049857</td>
      <td>-0.008278</td>
      <td>0.001212</td>
      <td>...</td>
      <td>0.007665</td>
      <td>NaN</td>
      <td>0.003432</td>
      <td>-0.002693</td>
      <td>-0.019359</td>
      <td>0.027627</td>
      <td>0.001458</td>
      <td>0.018007</td>
      <td>0.016194</td>
      <td>-0.004999</td>
    </tr>
    <tr>
      <th>HourlyRate</th>
      <td>0.024287</td>
      <td>0.023381</td>
      <td>0.031131</td>
      <td>0.016775</td>
      <td>NaN</td>
      <td>0.035179</td>
      <td>-0.049857</td>
      <td>1.000000</td>
      <td>0.042861</td>
      <td>-0.027853</td>
      <td>...</td>
      <td>0.001330</td>
      <td>NaN</td>
      <td>0.050263</td>
      <td>-0.002334</td>
      <td>-0.008548</td>
      <td>-0.004607</td>
      <td>-0.019582</td>
      <td>-0.024106</td>
      <td>-0.026716</td>
      <td>-0.020123</td>
    </tr>
    <tr>
      <th>JobInvolvement</th>
      <td>0.029820</td>
      <td>0.046135</td>
      <td>0.008783</td>
      <td>0.042438</td>
      <td>NaN</td>
      <td>-0.006888</td>
      <td>-0.008278</td>
      <td>0.042861</td>
      <td>1.000000</td>
      <td>-0.012630</td>
      <td>...</td>
      <td>0.034297</td>
      <td>NaN</td>
      <td>0.021523</td>
      <td>-0.005533</td>
      <td>-0.015338</td>
      <td>-0.014617</td>
      <td>-0.021355</td>
      <td>0.008717</td>
      <td>-0.024184</td>
      <td>0.025976</td>
    </tr>
    <tr>
      <th>JobLevel</th>
      <td>0.509604</td>
      <td>0.002966</td>
      <td>0.005303</td>
      <td>0.101589</td>
      <td>NaN</td>
      <td>-0.018519</td>
      <td>0.001212</td>
      <td>-0.027853</td>
      <td>-0.012630</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.021642</td>
      <td>NaN</td>
      <td>0.013984</td>
      <td>0.782208</td>
      <td>-0.018191</td>
      <td>0.037818</td>
      <td>0.534739</td>
      <td>0.389447</td>
      <td>0.353885</td>
      <td>0.375281</td>
    </tr>
    <tr>
      <th>JobSatisfaction</th>
      <td>-0.004892</td>
      <td>0.030571</td>
      <td>-0.003669</td>
      <td>-0.011296</td>
      <td>NaN</td>
      <td>-0.046247</td>
      <td>-0.006784</td>
      <td>-0.071335</td>
      <td>-0.021476</td>
      <td>-0.001944</td>
      <td>...</td>
      <td>-0.012454</td>
      <td>NaN</td>
      <td>0.010690</td>
      <td>-0.020185</td>
      <td>-0.005779</td>
      <td>-0.019459</td>
      <td>-0.003803</td>
      <td>-0.002305</td>
      <td>-0.018214</td>
      <td>-0.027656</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>0.497855</td>
      <td>0.007707</td>
      <td>-0.017014</td>
      <td>0.094961</td>
      <td>NaN</td>
      <td>-0.014829</td>
      <td>-0.006259</td>
      <td>-0.015794</td>
      <td>-0.015271</td>
      <td>0.950300</td>
      <td>...</td>
      <td>0.025873</td>
      <td>NaN</td>
      <td>0.005408</td>
      <td>0.772893</td>
      <td>-0.021736</td>
      <td>0.030683</td>
      <td>0.514285</td>
      <td>0.363818</td>
      <td>0.344978</td>
      <td>0.344079</td>
    </tr>
    <tr>
      <th>MonthlyRate</th>
      <td>0.028051</td>
      <td>-0.032182</td>
      <td>0.027473</td>
      <td>-0.026084</td>
      <td>NaN</td>
      <td>0.012648</td>
      <td>0.037600</td>
      <td>-0.015297</td>
      <td>-0.016322</td>
      <td>0.039563</td>
      <td>...</td>
      <td>-0.004085</td>
      <td>NaN</td>
      <td>-0.034323</td>
      <td>0.026442</td>
      <td>0.001467</td>
      <td>0.007963</td>
      <td>-0.023655</td>
      <td>-0.012815</td>
      <td>0.001567</td>
      <td>-0.036746</td>
    </tr>
    <tr>
      <th>NumCompaniesWorked</th>
      <td>0.299635</td>
      <td>0.038153</td>
      <td>-0.029251</td>
      <td>0.126317</td>
      <td>NaN</td>
      <td>-0.001251</td>
      <td>0.012594</td>
      <td>0.022157</td>
      <td>0.015012</td>
      <td>0.142501</td>
      <td>...</td>
      <td>0.052733</td>
      <td>NaN</td>
      <td>0.030075</td>
      <td>0.237639</td>
      <td>-0.066054</td>
      <td>-0.008366</td>
      <td>-0.118421</td>
      <td>-0.090754</td>
      <td>-0.036814</td>
      <td>-0.110319</td>
    </tr>
    <tr>
      <th>PercentSalaryHike</th>
      <td>0.003634</td>
      <td>0.022704</td>
      <td>0.040235</td>
      <td>-0.011111</td>
      <td>NaN</td>
      <td>-0.012944</td>
      <td>-0.031701</td>
      <td>-0.009062</td>
      <td>-0.017205</td>
      <td>-0.034730</td>
      <td>...</td>
      <td>-0.040490</td>
      <td>NaN</td>
      <td>0.007528</td>
      <td>-0.020608</td>
      <td>-0.005221</td>
      <td>-0.003280</td>
      <td>-0.035991</td>
      <td>-0.001520</td>
      <td>-0.022154</td>
      <td>-0.011985</td>
    </tr>
    <tr>
      <th>PerformanceRating</th>
      <td>0.001904</td>
      <td>0.000473</td>
      <td>0.027110</td>
      <td>-0.024539</td>
      <td>NaN</td>
      <td>-0.020359</td>
      <td>-0.029548</td>
      <td>-0.002172</td>
      <td>-0.029071</td>
      <td>-0.021222</td>
      <td>...</td>
      <td>-0.031351</td>
      <td>NaN</td>
      <td>0.003506</td>
      <td>0.006744</td>
      <td>-0.015579</td>
      <td>0.002572</td>
      <td>0.003435</td>
      <td>0.034986</td>
      <td>0.017896</td>
      <td>0.022827</td>
    </tr>
    <tr>
      <th>RelationshipSatisfaction</th>
      <td>0.053535</td>
      <td>0.007846</td>
      <td>0.006557</td>
      <td>-0.009118</td>
      <td>NaN</td>
      <td>-0.069861</td>
      <td>0.007665</td>
      <td>0.001330</td>
      <td>0.034297</td>
      <td>0.021642</td>
      <td>...</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.045952</td>
      <td>0.024054</td>
      <td>0.002497</td>
      <td>0.019604</td>
      <td>0.019367</td>
      <td>-0.015123</td>
      <td>0.033493</td>
      <td>-0.000867</td>
    </tr>
    <tr>
      <th>StandardHours</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>StockOptionLevel</th>
      <td>0.037510</td>
      <td>0.042143</td>
      <td>0.044872</td>
      <td>0.018422</td>
      <td>NaN</td>
      <td>0.062227</td>
      <td>0.003432</td>
      <td>0.050263</td>
      <td>0.021523</td>
      <td>0.013984</td>
      <td>...</td>
      <td>-0.045952</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.010136</td>
      <td>0.011274</td>
      <td>0.004129</td>
      <td>0.015058</td>
      <td>0.050818</td>
      <td>0.014352</td>
      <td>0.024698</td>
    </tr>
    <tr>
      <th>TotalWorkingYears</th>
      <td>0.680381</td>
      <td>0.014515</td>
      <td>0.004628</td>
      <td>0.148280</td>
      <td>NaN</td>
      <td>-0.014365</td>
      <td>-0.002693</td>
      <td>-0.002334</td>
      <td>-0.005533</td>
      <td>0.782208</td>
      <td>...</td>
      <td>0.024054</td>
      <td>NaN</td>
      <td>0.010136</td>
      <td>1.000000</td>
      <td>-0.035662</td>
      <td>0.001008</td>
      <td>0.628133</td>
      <td>0.460365</td>
      <td>0.404858</td>
      <td>0.459188</td>
    </tr>
    <tr>
      <th>TrainingTimesLastYear</th>
      <td>-0.019621</td>
      <td>0.002453</td>
      <td>-0.036942</td>
      <td>-0.025100</td>
      <td>NaN</td>
      <td>0.023603</td>
      <td>-0.019359</td>
      <td>-0.008548</td>
      <td>-0.015338</td>
      <td>-0.018191</td>
      <td>...</td>
      <td>0.002497</td>
      <td>NaN</td>
      <td>0.011274</td>
      <td>-0.035662</td>
      <td>1.000000</td>
      <td>0.028072</td>
      <td>0.003569</td>
      <td>-0.005738</td>
      <td>-0.002067</td>
      <td>-0.004096</td>
    </tr>
    <tr>
      <th>WorkLifeBalance</th>
      <td>-0.021490</td>
      <td>-0.037848</td>
      <td>-0.026556</td>
      <td>0.009819</td>
      <td>NaN</td>
      <td>0.010309</td>
      <td>0.027627</td>
      <td>-0.004607</td>
      <td>-0.014617</td>
      <td>0.037818</td>
      <td>...</td>
      <td>0.019604</td>
      <td>NaN</td>
      <td>0.004129</td>
      <td>0.001008</td>
      <td>0.028072</td>
      <td>1.000000</td>
      <td>0.012089</td>
      <td>0.049856</td>
      <td>0.008941</td>
      <td>0.002759</td>
    </tr>
    <tr>
      <th>YearsAtCompany</th>
      <td>0.311309</td>
      <td>-0.034055</td>
      <td>0.009508</td>
      <td>0.069114</td>
      <td>NaN</td>
      <td>-0.011240</td>
      <td>0.001458</td>
      <td>-0.019582</td>
      <td>-0.021355</td>
      <td>0.534739</td>
      <td>...</td>
      <td>0.019367</td>
      <td>NaN</td>
      <td>0.015058</td>
      <td>0.628133</td>
      <td>0.003569</td>
      <td>0.012089</td>
      <td>1.000000</td>
      <td>0.758754</td>
      <td>0.618409</td>
      <td>0.769212</td>
    </tr>
    <tr>
      <th>YearsInCurrentRole</th>
      <td>0.212901</td>
      <td>0.009932</td>
      <td>0.018845</td>
      <td>0.060236</td>
      <td>NaN</td>
      <td>-0.008416</td>
      <td>0.018007</td>
      <td>-0.024106</td>
      <td>0.008717</td>
      <td>0.389447</td>
      <td>...</td>
      <td>-0.015123</td>
      <td>NaN</td>
      <td>0.050818</td>
      <td>0.460365</td>
      <td>-0.005738</td>
      <td>0.049856</td>
      <td>0.758754</td>
      <td>1.000000</td>
      <td>0.548056</td>
      <td>0.714365</td>
    </tr>
    <tr>
      <th>YearsSinceLastPromotion</th>
      <td>0.216513</td>
      <td>-0.033229</td>
      <td>0.010029</td>
      <td>0.054254</td>
      <td>NaN</td>
      <td>-0.009019</td>
      <td>0.016194</td>
      <td>-0.026716</td>
      <td>-0.024184</td>
      <td>0.353885</td>
      <td>...</td>
      <td>0.033493</td>
      <td>NaN</td>
      <td>0.014352</td>
      <td>0.404858</td>
      <td>-0.002067</td>
      <td>0.008941</td>
      <td>0.618409</td>
      <td>0.548056</td>
      <td>1.000000</td>
      <td>0.510224</td>
    </tr>
    <tr>
      <th>YearsWithCurrManager</th>
      <td>0.202089</td>
      <td>-0.026363</td>
      <td>0.014406</td>
      <td>0.069065</td>
      <td>NaN</td>
      <td>-0.009197</td>
      <td>-0.004999</td>
      <td>-0.020123</td>
      <td>0.025976</td>
      <td>0.375281</td>
      <td>...</td>
      <td>-0.000867</td>
      <td>NaN</td>
      <td>0.024698</td>
      <td>0.459188</td>
      <td>-0.004096</td>
      <td>0.002759</td>
      <td>0.769212</td>
      <td>0.714365</td>
      <td>0.510224</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>26 rows × 26 columns</p>
</div>




![png](./templates/output_25_1.png)


***
#### Drop  columns that are not useful
***


```python
## Drop redundant columns
# 'DailyRate', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate' : check if they are correlated
# 'StandardHours' : all values are 80
# 'Over18' : All the employer above 18

col_to_drop = ['EmployeeCount','EmployeeNumber','Over18', 'StandardHours']
```


```python
hrdata.drop(['EmployeeCount','EmployeeNumber','Over18', 'StandardHours'], axis=1, inplace=True)

hrdata.columns

```




    Index(['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
           'DistanceFromHome', 'Education', 'EducationField',
           'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
           'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
           'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
           'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
           'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
           'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
           'YearsSinceLastPromotion', 'YearsWithCurrManager'],
          dtype='object')



***
#### Repeat the correlation matrix
***


```python
plt.figure(figsize=(10,8))
corr = hrdata.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels= corr.columns.values)
plt.title("Heatmap of Correlation Matrix of the IBM HR datasets")
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobSatisfaction</th>
      <th>MonthlyIncome</th>
      <th>...</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>0.010661</td>
      <td>-0.001686</td>
      <td>0.208034</td>
      <td>0.010146</td>
      <td>0.024287</td>
      <td>0.029820</td>
      <td>0.509604</td>
      <td>-0.004892</td>
      <td>0.497855</td>
      <td>...</td>
      <td>0.001904</td>
      <td>0.053535</td>
      <td>0.037510</td>
      <td>0.680381</td>
      <td>-0.019621</td>
      <td>-0.021490</td>
      <td>0.311309</td>
      <td>0.212901</td>
      <td>0.216513</td>
      <td>0.202089</td>
    </tr>
    <tr>
      <th>DailyRate</th>
      <td>0.010661</td>
      <td>1.000000</td>
      <td>-0.004985</td>
      <td>-0.016806</td>
      <td>0.018355</td>
      <td>0.023381</td>
      <td>0.046135</td>
      <td>0.002966</td>
      <td>0.030571</td>
      <td>0.007707</td>
      <td>...</td>
      <td>0.000473</td>
      <td>0.007846</td>
      <td>0.042143</td>
      <td>0.014515</td>
      <td>0.002453</td>
      <td>-0.037848</td>
      <td>-0.034055</td>
      <td>0.009932</td>
      <td>-0.033229</td>
      <td>-0.026363</td>
    </tr>
    <tr>
      <th>DistanceFromHome</th>
      <td>-0.001686</td>
      <td>-0.004985</td>
      <td>1.000000</td>
      <td>0.021042</td>
      <td>-0.016075</td>
      <td>0.031131</td>
      <td>0.008783</td>
      <td>0.005303</td>
      <td>-0.003669</td>
      <td>-0.017014</td>
      <td>...</td>
      <td>0.027110</td>
      <td>0.006557</td>
      <td>0.044872</td>
      <td>0.004628</td>
      <td>-0.036942</td>
      <td>-0.026556</td>
      <td>0.009508</td>
      <td>0.018845</td>
      <td>0.010029</td>
      <td>0.014406</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>0.208034</td>
      <td>-0.016806</td>
      <td>0.021042</td>
      <td>1.000000</td>
      <td>-0.027128</td>
      <td>0.016775</td>
      <td>0.042438</td>
      <td>0.101589</td>
      <td>-0.011296</td>
      <td>0.094961</td>
      <td>...</td>
      <td>-0.024539</td>
      <td>-0.009118</td>
      <td>0.018422</td>
      <td>0.148280</td>
      <td>-0.025100</td>
      <td>0.009819</td>
      <td>0.069114</td>
      <td>0.060236</td>
      <td>0.054254</td>
      <td>0.069065</td>
    </tr>
    <tr>
      <th>EnvironmentSatisfaction</th>
      <td>0.010146</td>
      <td>0.018355</td>
      <td>-0.016075</td>
      <td>-0.027128</td>
      <td>1.000000</td>
      <td>-0.049857</td>
      <td>-0.008278</td>
      <td>0.001212</td>
      <td>-0.006784</td>
      <td>-0.006259</td>
      <td>...</td>
      <td>-0.029548</td>
      <td>0.007665</td>
      <td>0.003432</td>
      <td>-0.002693</td>
      <td>-0.019359</td>
      <td>0.027627</td>
      <td>0.001458</td>
      <td>0.018007</td>
      <td>0.016194</td>
      <td>-0.004999</td>
    </tr>
    <tr>
      <th>HourlyRate</th>
      <td>0.024287</td>
      <td>0.023381</td>
      <td>0.031131</td>
      <td>0.016775</td>
      <td>-0.049857</td>
      <td>1.000000</td>
      <td>0.042861</td>
      <td>-0.027853</td>
      <td>-0.071335</td>
      <td>-0.015794</td>
      <td>...</td>
      <td>-0.002172</td>
      <td>0.001330</td>
      <td>0.050263</td>
      <td>-0.002334</td>
      <td>-0.008548</td>
      <td>-0.004607</td>
      <td>-0.019582</td>
      <td>-0.024106</td>
      <td>-0.026716</td>
      <td>-0.020123</td>
    </tr>
    <tr>
      <th>JobInvolvement</th>
      <td>0.029820</td>
      <td>0.046135</td>
      <td>0.008783</td>
      <td>0.042438</td>
      <td>-0.008278</td>
      <td>0.042861</td>
      <td>1.000000</td>
      <td>-0.012630</td>
      <td>-0.021476</td>
      <td>-0.015271</td>
      <td>...</td>
      <td>-0.029071</td>
      <td>0.034297</td>
      <td>0.021523</td>
      <td>-0.005533</td>
      <td>-0.015338</td>
      <td>-0.014617</td>
      <td>-0.021355</td>
      <td>0.008717</td>
      <td>-0.024184</td>
      <td>0.025976</td>
    </tr>
    <tr>
      <th>JobLevel</th>
      <td>0.509604</td>
      <td>0.002966</td>
      <td>0.005303</td>
      <td>0.101589</td>
      <td>0.001212</td>
      <td>-0.027853</td>
      <td>-0.012630</td>
      <td>1.000000</td>
      <td>-0.001944</td>
      <td>0.950300</td>
      <td>...</td>
      <td>-0.021222</td>
      <td>0.021642</td>
      <td>0.013984</td>
      <td>0.782208</td>
      <td>-0.018191</td>
      <td>0.037818</td>
      <td>0.534739</td>
      <td>0.389447</td>
      <td>0.353885</td>
      <td>0.375281</td>
    </tr>
    <tr>
      <th>JobSatisfaction</th>
      <td>-0.004892</td>
      <td>0.030571</td>
      <td>-0.003669</td>
      <td>-0.011296</td>
      <td>-0.006784</td>
      <td>-0.071335</td>
      <td>-0.021476</td>
      <td>-0.001944</td>
      <td>1.000000</td>
      <td>-0.007157</td>
      <td>...</td>
      <td>0.002297</td>
      <td>-0.012454</td>
      <td>0.010690</td>
      <td>-0.020185</td>
      <td>-0.005779</td>
      <td>-0.019459</td>
      <td>-0.003803</td>
      <td>-0.002305</td>
      <td>-0.018214</td>
      <td>-0.027656</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>0.497855</td>
      <td>0.007707</td>
      <td>-0.017014</td>
      <td>0.094961</td>
      <td>-0.006259</td>
      <td>-0.015794</td>
      <td>-0.015271</td>
      <td>0.950300</td>
      <td>-0.007157</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.017120</td>
      <td>0.025873</td>
      <td>0.005408</td>
      <td>0.772893</td>
      <td>-0.021736</td>
      <td>0.030683</td>
      <td>0.514285</td>
      <td>0.363818</td>
      <td>0.344978</td>
      <td>0.344079</td>
    </tr>
    <tr>
      <th>MonthlyRate</th>
      <td>0.028051</td>
      <td>-0.032182</td>
      <td>0.027473</td>
      <td>-0.026084</td>
      <td>0.037600</td>
      <td>-0.015297</td>
      <td>-0.016322</td>
      <td>0.039563</td>
      <td>0.000644</td>
      <td>0.034814</td>
      <td>...</td>
      <td>-0.009811</td>
      <td>-0.004085</td>
      <td>-0.034323</td>
      <td>0.026442</td>
      <td>0.001467</td>
      <td>0.007963</td>
      <td>-0.023655</td>
      <td>-0.012815</td>
      <td>0.001567</td>
      <td>-0.036746</td>
    </tr>
    <tr>
      <th>NumCompaniesWorked</th>
      <td>0.299635</td>
      <td>0.038153</td>
      <td>-0.029251</td>
      <td>0.126317</td>
      <td>0.012594</td>
      <td>0.022157</td>
      <td>0.015012</td>
      <td>0.142501</td>
      <td>-0.055699</td>
      <td>0.149515</td>
      <td>...</td>
      <td>-0.014095</td>
      <td>0.052733</td>
      <td>0.030075</td>
      <td>0.237639</td>
      <td>-0.066054</td>
      <td>-0.008366</td>
      <td>-0.118421</td>
      <td>-0.090754</td>
      <td>-0.036814</td>
      <td>-0.110319</td>
    </tr>
    <tr>
      <th>PercentSalaryHike</th>
      <td>0.003634</td>
      <td>0.022704</td>
      <td>0.040235</td>
      <td>-0.011111</td>
      <td>-0.031701</td>
      <td>-0.009062</td>
      <td>-0.017205</td>
      <td>-0.034730</td>
      <td>0.020002</td>
      <td>-0.027269</td>
      <td>...</td>
      <td>0.773550</td>
      <td>-0.040490</td>
      <td>0.007528</td>
      <td>-0.020608</td>
      <td>-0.005221</td>
      <td>-0.003280</td>
      <td>-0.035991</td>
      <td>-0.001520</td>
      <td>-0.022154</td>
      <td>-0.011985</td>
    </tr>
    <tr>
      <th>PerformanceRating</th>
      <td>0.001904</td>
      <td>0.000473</td>
      <td>0.027110</td>
      <td>-0.024539</td>
      <td>-0.029548</td>
      <td>-0.002172</td>
      <td>-0.029071</td>
      <td>-0.021222</td>
      <td>0.002297</td>
      <td>-0.017120</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.031351</td>
      <td>0.003506</td>
      <td>0.006744</td>
      <td>-0.015579</td>
      <td>0.002572</td>
      <td>0.003435</td>
      <td>0.034986</td>
      <td>0.017896</td>
      <td>0.022827</td>
    </tr>
    <tr>
      <th>RelationshipSatisfaction</th>
      <td>0.053535</td>
      <td>0.007846</td>
      <td>0.006557</td>
      <td>-0.009118</td>
      <td>0.007665</td>
      <td>0.001330</td>
      <td>0.034297</td>
      <td>0.021642</td>
      <td>-0.012454</td>
      <td>0.025873</td>
      <td>...</td>
      <td>-0.031351</td>
      <td>1.000000</td>
      <td>-0.045952</td>
      <td>0.024054</td>
      <td>0.002497</td>
      <td>0.019604</td>
      <td>0.019367</td>
      <td>-0.015123</td>
      <td>0.033493</td>
      <td>-0.000867</td>
    </tr>
    <tr>
      <th>StockOptionLevel</th>
      <td>0.037510</td>
      <td>0.042143</td>
      <td>0.044872</td>
      <td>0.018422</td>
      <td>0.003432</td>
      <td>0.050263</td>
      <td>0.021523</td>
      <td>0.013984</td>
      <td>0.010690</td>
      <td>0.005408</td>
      <td>...</td>
      <td>0.003506</td>
      <td>-0.045952</td>
      <td>1.000000</td>
      <td>0.010136</td>
      <td>0.011274</td>
      <td>0.004129</td>
      <td>0.015058</td>
      <td>0.050818</td>
      <td>0.014352</td>
      <td>0.024698</td>
    </tr>
    <tr>
      <th>TotalWorkingYears</th>
      <td>0.680381</td>
      <td>0.014515</td>
      <td>0.004628</td>
      <td>0.148280</td>
      <td>-0.002693</td>
      <td>-0.002334</td>
      <td>-0.005533</td>
      <td>0.782208</td>
      <td>-0.020185</td>
      <td>0.772893</td>
      <td>...</td>
      <td>0.006744</td>
      <td>0.024054</td>
      <td>0.010136</td>
      <td>1.000000</td>
      <td>-0.035662</td>
      <td>0.001008</td>
      <td>0.628133</td>
      <td>0.460365</td>
      <td>0.404858</td>
      <td>0.459188</td>
    </tr>
    <tr>
      <th>TrainingTimesLastYear</th>
      <td>-0.019621</td>
      <td>0.002453</td>
      <td>-0.036942</td>
      <td>-0.025100</td>
      <td>-0.019359</td>
      <td>-0.008548</td>
      <td>-0.015338</td>
      <td>-0.018191</td>
      <td>-0.005779</td>
      <td>-0.021736</td>
      <td>...</td>
      <td>-0.015579</td>
      <td>0.002497</td>
      <td>0.011274</td>
      <td>-0.035662</td>
      <td>1.000000</td>
      <td>0.028072</td>
      <td>0.003569</td>
      <td>-0.005738</td>
      <td>-0.002067</td>
      <td>-0.004096</td>
    </tr>
    <tr>
      <th>WorkLifeBalance</th>
      <td>-0.021490</td>
      <td>-0.037848</td>
      <td>-0.026556</td>
      <td>0.009819</td>
      <td>0.027627</td>
      <td>-0.004607</td>
      <td>-0.014617</td>
      <td>0.037818</td>
      <td>-0.019459</td>
      <td>0.030683</td>
      <td>...</td>
      <td>0.002572</td>
      <td>0.019604</td>
      <td>0.004129</td>
      <td>0.001008</td>
      <td>0.028072</td>
      <td>1.000000</td>
      <td>0.012089</td>
      <td>0.049856</td>
      <td>0.008941</td>
      <td>0.002759</td>
    </tr>
    <tr>
      <th>YearsAtCompany</th>
      <td>0.311309</td>
      <td>-0.034055</td>
      <td>0.009508</td>
      <td>0.069114</td>
      <td>0.001458</td>
      <td>-0.019582</td>
      <td>-0.021355</td>
      <td>0.534739</td>
      <td>-0.003803</td>
      <td>0.514285</td>
      <td>...</td>
      <td>0.003435</td>
      <td>0.019367</td>
      <td>0.015058</td>
      <td>0.628133</td>
      <td>0.003569</td>
      <td>0.012089</td>
      <td>1.000000</td>
      <td>0.758754</td>
      <td>0.618409</td>
      <td>0.769212</td>
    </tr>
    <tr>
      <th>YearsInCurrentRole</th>
      <td>0.212901</td>
      <td>0.009932</td>
      <td>0.018845</td>
      <td>0.060236</td>
      <td>0.018007</td>
      <td>-0.024106</td>
      <td>0.008717</td>
      <td>0.389447</td>
      <td>-0.002305</td>
      <td>0.363818</td>
      <td>...</td>
      <td>0.034986</td>
      <td>-0.015123</td>
      <td>0.050818</td>
      <td>0.460365</td>
      <td>-0.005738</td>
      <td>0.049856</td>
      <td>0.758754</td>
      <td>1.000000</td>
      <td>0.548056</td>
      <td>0.714365</td>
    </tr>
    <tr>
      <th>YearsSinceLastPromotion</th>
      <td>0.216513</td>
      <td>-0.033229</td>
      <td>0.010029</td>
      <td>0.054254</td>
      <td>0.016194</td>
      <td>-0.026716</td>
      <td>-0.024184</td>
      <td>0.353885</td>
      <td>-0.018214</td>
      <td>0.344978</td>
      <td>...</td>
      <td>0.017896</td>
      <td>0.033493</td>
      <td>0.014352</td>
      <td>0.404858</td>
      <td>-0.002067</td>
      <td>0.008941</td>
      <td>0.618409</td>
      <td>0.548056</td>
      <td>1.000000</td>
      <td>0.510224</td>
    </tr>
    <tr>
      <th>YearsWithCurrManager</th>
      <td>0.202089</td>
      <td>-0.026363</td>
      <td>0.014406</td>
      <td>0.069065</td>
      <td>-0.004999</td>
      <td>-0.020123</td>
      <td>0.025976</td>
      <td>0.375281</td>
      <td>-0.027656</td>
      <td>0.344079</td>
      <td>...</td>
      <td>0.022827</td>
      <td>-0.000867</td>
      <td>0.024698</td>
      <td>0.459188</td>
      <td>-0.004096</td>
      <td>0.002759</td>
      <td>0.769212</td>
      <td>0.714365</td>
      <td>0.510224</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>23 rows × 23 columns</p>
</div>




![png](./templates/output_30_1.png)


***
#### Change Attrition value to 0 : NO, 1 :YES

This will allow us to view the correlation between Attrition and other variableS
***


```python
hrdata['Attrition'] = hrdata.Attrition.map(dict(Yes=1, No=0))
hrdata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>...</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>1</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>2</td>
      <td>Female</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>0</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>3</td>
      <td>Male</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>4</td>
      <td>Male</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>0</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>4</td>
      <td>Female</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>Male</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



***
#### Correlation Matrix 3
***


```python
plt.figure(figsize=(10,8))
corr = hrdata.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels= corr.columns.values)
plt.title("Heatmap of Correlation Matrix of the IBM HR datasets")
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobSatisfaction</th>
      <th>...</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>-0.159205</td>
      <td>0.010661</td>
      <td>-0.001686</td>
      <td>0.208034</td>
      <td>0.010146</td>
      <td>0.024287</td>
      <td>0.029820</td>
      <td>0.509604</td>
      <td>-0.004892</td>
      <td>...</td>
      <td>0.001904</td>
      <td>0.053535</td>
      <td>0.037510</td>
      <td>0.680381</td>
      <td>-0.019621</td>
      <td>-0.021490</td>
      <td>0.311309</td>
      <td>0.212901</td>
      <td>0.216513</td>
      <td>0.202089</td>
    </tr>
    <tr>
      <th>Attrition</th>
      <td>-0.159205</td>
      <td>1.000000</td>
      <td>-0.056652</td>
      <td>0.077924</td>
      <td>-0.031373</td>
      <td>-0.103369</td>
      <td>-0.006846</td>
      <td>-0.130016</td>
      <td>-0.169105</td>
      <td>-0.103481</td>
      <td>...</td>
      <td>0.002889</td>
      <td>-0.045872</td>
      <td>-0.137145</td>
      <td>-0.171063</td>
      <td>-0.059478</td>
      <td>-0.063939</td>
      <td>-0.134392</td>
      <td>-0.160545</td>
      <td>-0.033019</td>
      <td>-0.156199</td>
    </tr>
    <tr>
      <th>DailyRate</th>
      <td>0.010661</td>
      <td>-0.056652</td>
      <td>1.000000</td>
      <td>-0.004985</td>
      <td>-0.016806</td>
      <td>0.018355</td>
      <td>0.023381</td>
      <td>0.046135</td>
      <td>0.002966</td>
      <td>0.030571</td>
      <td>...</td>
      <td>0.000473</td>
      <td>0.007846</td>
      <td>0.042143</td>
      <td>0.014515</td>
      <td>0.002453</td>
      <td>-0.037848</td>
      <td>-0.034055</td>
      <td>0.009932</td>
      <td>-0.033229</td>
      <td>-0.026363</td>
    </tr>
    <tr>
      <th>DistanceFromHome</th>
      <td>-0.001686</td>
      <td>0.077924</td>
      <td>-0.004985</td>
      <td>1.000000</td>
      <td>0.021042</td>
      <td>-0.016075</td>
      <td>0.031131</td>
      <td>0.008783</td>
      <td>0.005303</td>
      <td>-0.003669</td>
      <td>...</td>
      <td>0.027110</td>
      <td>0.006557</td>
      <td>0.044872</td>
      <td>0.004628</td>
      <td>-0.036942</td>
      <td>-0.026556</td>
      <td>0.009508</td>
      <td>0.018845</td>
      <td>0.010029</td>
      <td>0.014406</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>0.208034</td>
      <td>-0.031373</td>
      <td>-0.016806</td>
      <td>0.021042</td>
      <td>1.000000</td>
      <td>-0.027128</td>
      <td>0.016775</td>
      <td>0.042438</td>
      <td>0.101589</td>
      <td>-0.011296</td>
      <td>...</td>
      <td>-0.024539</td>
      <td>-0.009118</td>
      <td>0.018422</td>
      <td>0.148280</td>
      <td>-0.025100</td>
      <td>0.009819</td>
      <td>0.069114</td>
      <td>0.060236</td>
      <td>0.054254</td>
      <td>0.069065</td>
    </tr>
    <tr>
      <th>EnvironmentSatisfaction</th>
      <td>0.010146</td>
      <td>-0.103369</td>
      <td>0.018355</td>
      <td>-0.016075</td>
      <td>-0.027128</td>
      <td>1.000000</td>
      <td>-0.049857</td>
      <td>-0.008278</td>
      <td>0.001212</td>
      <td>-0.006784</td>
      <td>...</td>
      <td>-0.029548</td>
      <td>0.007665</td>
      <td>0.003432</td>
      <td>-0.002693</td>
      <td>-0.019359</td>
      <td>0.027627</td>
      <td>0.001458</td>
      <td>0.018007</td>
      <td>0.016194</td>
      <td>-0.004999</td>
    </tr>
    <tr>
      <th>HourlyRate</th>
      <td>0.024287</td>
      <td>-0.006846</td>
      <td>0.023381</td>
      <td>0.031131</td>
      <td>0.016775</td>
      <td>-0.049857</td>
      <td>1.000000</td>
      <td>0.042861</td>
      <td>-0.027853</td>
      <td>-0.071335</td>
      <td>...</td>
      <td>-0.002172</td>
      <td>0.001330</td>
      <td>0.050263</td>
      <td>-0.002334</td>
      <td>-0.008548</td>
      <td>-0.004607</td>
      <td>-0.019582</td>
      <td>-0.024106</td>
      <td>-0.026716</td>
      <td>-0.020123</td>
    </tr>
    <tr>
      <th>JobInvolvement</th>
      <td>0.029820</td>
      <td>-0.130016</td>
      <td>0.046135</td>
      <td>0.008783</td>
      <td>0.042438</td>
      <td>-0.008278</td>
      <td>0.042861</td>
      <td>1.000000</td>
      <td>-0.012630</td>
      <td>-0.021476</td>
      <td>...</td>
      <td>-0.029071</td>
      <td>0.034297</td>
      <td>0.021523</td>
      <td>-0.005533</td>
      <td>-0.015338</td>
      <td>-0.014617</td>
      <td>-0.021355</td>
      <td>0.008717</td>
      <td>-0.024184</td>
      <td>0.025976</td>
    </tr>
    <tr>
      <th>JobLevel</th>
      <td>0.509604</td>
      <td>-0.169105</td>
      <td>0.002966</td>
      <td>0.005303</td>
      <td>0.101589</td>
      <td>0.001212</td>
      <td>-0.027853</td>
      <td>-0.012630</td>
      <td>1.000000</td>
      <td>-0.001944</td>
      <td>...</td>
      <td>-0.021222</td>
      <td>0.021642</td>
      <td>0.013984</td>
      <td>0.782208</td>
      <td>-0.018191</td>
      <td>0.037818</td>
      <td>0.534739</td>
      <td>0.389447</td>
      <td>0.353885</td>
      <td>0.375281</td>
    </tr>
    <tr>
      <th>JobSatisfaction</th>
      <td>-0.004892</td>
      <td>-0.103481</td>
      <td>0.030571</td>
      <td>-0.003669</td>
      <td>-0.011296</td>
      <td>-0.006784</td>
      <td>-0.071335</td>
      <td>-0.021476</td>
      <td>-0.001944</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.002297</td>
      <td>-0.012454</td>
      <td>0.010690</td>
      <td>-0.020185</td>
      <td>-0.005779</td>
      <td>-0.019459</td>
      <td>-0.003803</td>
      <td>-0.002305</td>
      <td>-0.018214</td>
      <td>-0.027656</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>0.497855</td>
      <td>-0.159840</td>
      <td>0.007707</td>
      <td>-0.017014</td>
      <td>0.094961</td>
      <td>-0.006259</td>
      <td>-0.015794</td>
      <td>-0.015271</td>
      <td>0.950300</td>
      <td>-0.007157</td>
      <td>...</td>
      <td>-0.017120</td>
      <td>0.025873</td>
      <td>0.005408</td>
      <td>0.772893</td>
      <td>-0.021736</td>
      <td>0.030683</td>
      <td>0.514285</td>
      <td>0.363818</td>
      <td>0.344978</td>
      <td>0.344079</td>
    </tr>
    <tr>
      <th>MonthlyRate</th>
      <td>0.028051</td>
      <td>0.015170</td>
      <td>-0.032182</td>
      <td>0.027473</td>
      <td>-0.026084</td>
      <td>0.037600</td>
      <td>-0.015297</td>
      <td>-0.016322</td>
      <td>0.039563</td>
      <td>0.000644</td>
      <td>...</td>
      <td>-0.009811</td>
      <td>-0.004085</td>
      <td>-0.034323</td>
      <td>0.026442</td>
      <td>0.001467</td>
      <td>0.007963</td>
      <td>-0.023655</td>
      <td>-0.012815</td>
      <td>0.001567</td>
      <td>-0.036746</td>
    </tr>
    <tr>
      <th>NumCompaniesWorked</th>
      <td>0.299635</td>
      <td>0.043494</td>
      <td>0.038153</td>
      <td>-0.029251</td>
      <td>0.126317</td>
      <td>0.012594</td>
      <td>0.022157</td>
      <td>0.015012</td>
      <td>0.142501</td>
      <td>-0.055699</td>
      <td>...</td>
      <td>-0.014095</td>
      <td>0.052733</td>
      <td>0.030075</td>
      <td>0.237639</td>
      <td>-0.066054</td>
      <td>-0.008366</td>
      <td>-0.118421</td>
      <td>-0.090754</td>
      <td>-0.036814</td>
      <td>-0.110319</td>
    </tr>
    <tr>
      <th>PercentSalaryHike</th>
      <td>0.003634</td>
      <td>-0.013478</td>
      <td>0.022704</td>
      <td>0.040235</td>
      <td>-0.011111</td>
      <td>-0.031701</td>
      <td>-0.009062</td>
      <td>-0.017205</td>
      <td>-0.034730</td>
      <td>0.020002</td>
      <td>...</td>
      <td>0.773550</td>
      <td>-0.040490</td>
      <td>0.007528</td>
      <td>-0.020608</td>
      <td>-0.005221</td>
      <td>-0.003280</td>
      <td>-0.035991</td>
      <td>-0.001520</td>
      <td>-0.022154</td>
      <td>-0.011985</td>
    </tr>
    <tr>
      <th>PerformanceRating</th>
      <td>0.001904</td>
      <td>0.002889</td>
      <td>0.000473</td>
      <td>0.027110</td>
      <td>-0.024539</td>
      <td>-0.029548</td>
      <td>-0.002172</td>
      <td>-0.029071</td>
      <td>-0.021222</td>
      <td>0.002297</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.031351</td>
      <td>0.003506</td>
      <td>0.006744</td>
      <td>-0.015579</td>
      <td>0.002572</td>
      <td>0.003435</td>
      <td>0.034986</td>
      <td>0.017896</td>
      <td>0.022827</td>
    </tr>
    <tr>
      <th>RelationshipSatisfaction</th>
      <td>0.053535</td>
      <td>-0.045872</td>
      <td>0.007846</td>
      <td>0.006557</td>
      <td>-0.009118</td>
      <td>0.007665</td>
      <td>0.001330</td>
      <td>0.034297</td>
      <td>0.021642</td>
      <td>-0.012454</td>
      <td>...</td>
      <td>-0.031351</td>
      <td>1.000000</td>
      <td>-0.045952</td>
      <td>0.024054</td>
      <td>0.002497</td>
      <td>0.019604</td>
      <td>0.019367</td>
      <td>-0.015123</td>
      <td>0.033493</td>
      <td>-0.000867</td>
    </tr>
    <tr>
      <th>StockOptionLevel</th>
      <td>0.037510</td>
      <td>-0.137145</td>
      <td>0.042143</td>
      <td>0.044872</td>
      <td>0.018422</td>
      <td>0.003432</td>
      <td>0.050263</td>
      <td>0.021523</td>
      <td>0.013984</td>
      <td>0.010690</td>
      <td>...</td>
      <td>0.003506</td>
      <td>-0.045952</td>
      <td>1.000000</td>
      <td>0.010136</td>
      <td>0.011274</td>
      <td>0.004129</td>
      <td>0.015058</td>
      <td>0.050818</td>
      <td>0.014352</td>
      <td>0.024698</td>
    </tr>
    <tr>
      <th>TotalWorkingYears</th>
      <td>0.680381</td>
      <td>-0.171063</td>
      <td>0.014515</td>
      <td>0.004628</td>
      <td>0.148280</td>
      <td>-0.002693</td>
      <td>-0.002334</td>
      <td>-0.005533</td>
      <td>0.782208</td>
      <td>-0.020185</td>
      <td>...</td>
      <td>0.006744</td>
      <td>0.024054</td>
      <td>0.010136</td>
      <td>1.000000</td>
      <td>-0.035662</td>
      <td>0.001008</td>
      <td>0.628133</td>
      <td>0.460365</td>
      <td>0.404858</td>
      <td>0.459188</td>
    </tr>
    <tr>
      <th>TrainingTimesLastYear</th>
      <td>-0.019621</td>
      <td>-0.059478</td>
      <td>0.002453</td>
      <td>-0.036942</td>
      <td>-0.025100</td>
      <td>-0.019359</td>
      <td>-0.008548</td>
      <td>-0.015338</td>
      <td>-0.018191</td>
      <td>-0.005779</td>
      <td>...</td>
      <td>-0.015579</td>
      <td>0.002497</td>
      <td>0.011274</td>
      <td>-0.035662</td>
      <td>1.000000</td>
      <td>0.028072</td>
      <td>0.003569</td>
      <td>-0.005738</td>
      <td>-0.002067</td>
      <td>-0.004096</td>
    </tr>
    <tr>
      <th>WorkLifeBalance</th>
      <td>-0.021490</td>
      <td>-0.063939</td>
      <td>-0.037848</td>
      <td>-0.026556</td>
      <td>0.009819</td>
      <td>0.027627</td>
      <td>-0.004607</td>
      <td>-0.014617</td>
      <td>0.037818</td>
      <td>-0.019459</td>
      <td>...</td>
      <td>0.002572</td>
      <td>0.019604</td>
      <td>0.004129</td>
      <td>0.001008</td>
      <td>0.028072</td>
      <td>1.000000</td>
      <td>0.012089</td>
      <td>0.049856</td>
      <td>0.008941</td>
      <td>0.002759</td>
    </tr>
    <tr>
      <th>YearsAtCompany</th>
      <td>0.311309</td>
      <td>-0.134392</td>
      <td>-0.034055</td>
      <td>0.009508</td>
      <td>0.069114</td>
      <td>0.001458</td>
      <td>-0.019582</td>
      <td>-0.021355</td>
      <td>0.534739</td>
      <td>-0.003803</td>
      <td>...</td>
      <td>0.003435</td>
      <td>0.019367</td>
      <td>0.015058</td>
      <td>0.628133</td>
      <td>0.003569</td>
      <td>0.012089</td>
      <td>1.000000</td>
      <td>0.758754</td>
      <td>0.618409</td>
      <td>0.769212</td>
    </tr>
    <tr>
      <th>YearsInCurrentRole</th>
      <td>0.212901</td>
      <td>-0.160545</td>
      <td>0.009932</td>
      <td>0.018845</td>
      <td>0.060236</td>
      <td>0.018007</td>
      <td>-0.024106</td>
      <td>0.008717</td>
      <td>0.389447</td>
      <td>-0.002305</td>
      <td>...</td>
      <td>0.034986</td>
      <td>-0.015123</td>
      <td>0.050818</td>
      <td>0.460365</td>
      <td>-0.005738</td>
      <td>0.049856</td>
      <td>0.758754</td>
      <td>1.000000</td>
      <td>0.548056</td>
      <td>0.714365</td>
    </tr>
    <tr>
      <th>YearsSinceLastPromotion</th>
      <td>0.216513</td>
      <td>-0.033019</td>
      <td>-0.033229</td>
      <td>0.010029</td>
      <td>0.054254</td>
      <td>0.016194</td>
      <td>-0.026716</td>
      <td>-0.024184</td>
      <td>0.353885</td>
      <td>-0.018214</td>
      <td>...</td>
      <td>0.017896</td>
      <td>0.033493</td>
      <td>0.014352</td>
      <td>0.404858</td>
      <td>-0.002067</td>
      <td>0.008941</td>
      <td>0.618409</td>
      <td>0.548056</td>
      <td>1.000000</td>
      <td>0.510224</td>
    </tr>
    <tr>
      <th>YearsWithCurrManager</th>
      <td>0.202089</td>
      <td>-0.156199</td>
      <td>-0.026363</td>
      <td>0.014406</td>
      <td>0.069065</td>
      <td>-0.004999</td>
      <td>-0.020123</td>
      <td>0.025976</td>
      <td>0.375281</td>
      <td>-0.027656</td>
      <td>...</td>
      <td>0.022827</td>
      <td>-0.000867</td>
      <td>0.024698</td>
      <td>0.459188</td>
      <td>-0.004096</td>
      <td>0.002759</td>
      <td>0.769212</td>
      <td>0.714365</td>
      <td>0.510224</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>24 rows × 24 columns</p>
</div>




![png](./templates/output_34_1.png)


***
#### Distribution of Age, MonthlyIncome, JobSatisfaction
***


```python

fig, axes = plt.subplots(ncols=3, figsize=(15, 6))

# Age graph
sns.distplot(hrdata.Age, kde=True, color="g", ax=axes[0]).set_title('Employee Age Distribution')
axes[0].set_ylabel('Employee Count')

# MonthlyIncome graph
sns.distplot(hrdata.MonthlyIncome, kde=True, color='r', ax=axes[1]).set_title('Employee Monthly Income Distribution')
axes[1].set_ylabel('Employee Count')

# JobSatisfaction graph
sns.distplot(hrdata.JobSatisfaction, kde=True, color='b', ax=axes[2]).set_title('Employee Job satisfaction Distribution')
axes[2].set_ylabel('Employee Count')

```




    Text(0, 0.5, 'Employee Count')




![png](./templates/output_36_1.png)


***
#### Distribution of DailyRate, Education and DistanceFromeHome
***


```python
## plot the distribution of DailyRate, Education and DiatanceFromeHome

# setup matplotlib figure

fig, axes = plt.subplots(ncols=3, figsize=(15, 6))

# DailyRate graph
sns.distplot(hrdata.DailyRate, kde=True, color="g", ax=axes[0]).set_title('Employee Daily Rate Distribution')
axes[0].set_ylabel('Employee Count')

# Education graph
sns.distplot(hrdata.Education, kde=True, color='r', ax=axes[1]).set_title('Employee Education Level Distribution')
axes[1].set_ylabel('Employee Count')

# Distance From Home graph
sns.distplot(hrdata.DistanceFromHome, kde=True, color='b', ax=axes[2]).set_title('Employee Distance From Home Distribution')
axes[2].set_ylabel('Employee Count')
```




    Text(0, 0.5, 'Employee Count')




![png](./templates/output_38_1.png)


***
#### Age and Monthly Income
***


```python
sns.lmplot(x='Age', y='MonthlyIncome', data=hrdata,fit_reg=True,hue='Attrition') 
```




    <seaborn.axisgrid.FacetGrid at 0x13820a9e588>




![png](./templates/output_40_1.png)


***
* Employee with low monthly income and withing the Age of 18 to 35 are more likely to leave the company
***

***
#### Monthly Income and Job satisfaction
***


```python
sns.lmplot(x='MonthlyIncome', y='JobSatisfaction', data=hrdata, fit_reg=False, hue='Attrition')
```




    <seaborn.axisgrid.FacetGrid at 0x13820f33788>




![png](./templates/output_43_1.png)


***
* Employee with low monthly income inrespective of job Satisfaction are likely to leave the company
***

***
#### Job satisfaction and distance from home

Is the diatance a factor of job satisfation?
***


```python
sns.lmplot(x='DistanceFromHome', y='JobSatisfaction', data=hrdata, fit_reg=False, hue='Attrition')
```




    <seaborn.axisgrid.FacetGrid at 0x138217edf88>




![png](./templates/output_46_1.png)


***
#### Age and Distance from home
***


```python
sns.lmplot(x='Age', y='JobSatisfaction', data=hrdata, fit_reg=False, hue='Attrition')
```




    <seaborn.axisgrid.FacetGrid at 0x13821867e48>




![png](./templates/output_48_1.png)


***
#### Performance rating and Job satisfaction
***


```python
sns.lmplot(x='PerformanceRating', y='JobSatisfaction', data=hrdata, fit_reg=False, hue='Attrition')
```




    <seaborn.axisgrid.FacetGrid at 0x138218fa448>




![png](./templates/output_50_1.png)


***
#### Employee JobSatisfaction
***


```python
#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(hrdata.loc[(hrdata['Attrition'] == 0),'JobSatisfaction'] , color='b',shade=True, label='No Attrition')
ax=sns.kdeplot(hrdata.loc[(hrdata['Attrition'] == 1),'JobSatisfaction'] , color='r',shade=True, label='Attrition')
plt.title('Employee Job Satisfaction Distribution - Attrition V.S. No Attrition')
```




    Text(0.5, 1.0, 'Employee Job Satisfaction Distribution - Attrition V.S. No Attrition')




![png](./templates/output_52_1.png)


***
* Employee with lower job satisfation are likely to leave compared to employee with higher job satisfation
***

***
#### Employee Age
***


```python
#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(hrdata.loc[(hrdata['Attrition'] == 0),'Age'] , color='b',shade=True, label='No Attrition')
ax=sns.kdeplot(hrdata.loc[(hrdata['Attrition'] == 1),'Age'] , color='r',shade=True, label='Attrition')
plt.title('Employee Age Distribution - Attrition V.S. No Attrition')
```




    Text(0.5, 1.0, 'Employee Age Distribution - Attrition V.S. No Attrition')




![png](./templates/output_55_1.png)


***
#### Monthly Income
***


```python
#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(hrdata.loc[(hrdata['Attrition'] == 0),'MonthlyIncome'] , color='b',shade=True, label='No Attrition')
ax=sns.kdeplot(hrdata.loc[(hrdata['Attrition'] == 1),'MonthlyIncome'] , color='r',shade=True, label='Attrition')
plt.title('Employee Monthly Income Distribution - Attrition V.S. No Attrition')
```




    Text(0.5, 1.0, 'Employee Monthly Income Distribution - Attrition V.S. No Attrition')




![png](./templates/output_57_1.png)


***
* Employee with low Monthly Income are likely to leave the company
***

***
#### Employee Monthly Income Distribution based on Overtime
***


```python
ax = sns.barplot(x="OverTime", y="MonthlyIncome", hue="Attrition", data=hrdata, estimator=lambda x: len(x) / len(hrdata) * 100)
ax.set(ylabel="Percent")
```




    [Text(0, 0.5, 'Percent')]




![png](./templates/output_60_1.png)


***
#### Department Distribution
***


```python
hrleft = hrdata[hrdata['Attrition']==1]

hrleft = pd.DataFrame(hrleft.Department.value_counts()).reset_index()
hrstay = pd.DataFrame(hrdata.Department.value_counts()).reset_index()

hr_merge = pd.merge(hrleft, hrstay, how='inner', on='index')

hr_merge = hr_merge.rename(columns={"Department_x":'left', "Department_y":'stay', "index":'Department' })
hr_merge
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Department</th>
      <th>left</th>
      <th>stay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Research &amp; Development</td>
      <td>133</td>
      <td>961</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sales</td>
      <td>92</td>
      <td>446</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Human Resources</td>
      <td>12</td>
      <td>63</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))

# Plot the total schools per city
sns.set_color_codes("pastel")
sns.barplot(x="stay", y='Department', data=hr_merge,
            label="Total", color="b")

# Plot the total community schools per city
sns.set_color_codes("muted")
sns.barplot(x="left", y="Department", data=hr_merge,
            label="Left", color="r")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set( ylabel="Department", title='Employees Per Department',
       xlabel="# of Employees")
sns.despine(left=True, bottom=True)
```


![png](./templates/output_63_0.png)


***
### KMean Clustering of the Employee Attrition
***


```python
# Import KMeans Model
from sklearn.cluster import KMeans

# Graph and create 3 clusters of Employee Turnover
kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(hrdata[hrdata.Attrition==1][["MonthlyIncome","JobSatisfaction"]])

kmeans_colors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_]

fig = plt.figure(figsize=(10, 6))
plt.scatter(x="JobSatisfaction",y="MonthlyIncome", data=hrdata[hrdata.Attrition==1],alpha=0.25,color = kmeans_colors)
plt.xlabel("Job Satisfaction")
plt.ylabel("Monthly Income")
plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)
plt.title("Clusters of Employee Attrition")
plt.show()
```


![png](./templates/output_65_0.png)


***
### Data Pre-Processing
* Seperate Categrical and numeric variable, apply get_dummies() to categorical variable, then combined.
***


```python
hrdata.dtypes
```




    Age                          int64
    Attrition                    int64
    BusinessTravel              object
    DailyRate                    int64
    Department                  object
    DistanceFromHome             int64
    Education                    int64
    EducationField              object
    EnvironmentSatisfaction      int64
    Gender                      object
    HourlyRate                   int64
    JobInvolvement               int64
    JobLevel                     int64
    JobRole                     object
    JobSatisfaction              int64
    MaritalStatus               object
    MonthlyIncome                int64
    MonthlyRate                  int64
    NumCompaniesWorked           int64
    OverTime                    object
    PercentSalaryHike            int64
    PerformanceRating            int64
    RelationshipSatisfaction     int64
    StockOptionLevel             int64
    TotalWorkingYears            int64
    TrainingTimesLastYear        int64
    WorkLifeBalance              int64
    YearsAtCompany               int64
    YearsInCurrentRole           int64
    YearsSinceLastPromotion      int64
    YearsWithCurrManager         int64
    dtype: object




```python
cat_variables = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']
num_variables = ['Age', 'Attrition', 'DailyRate','DistanceFromHome', 'Education','EnvironmentSatisfaction', 'HourlyRate',
                 'JobInvolvement','JobLevel', 'JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked','PercentSalaryHike', 
                 'PerformanceRating','RelationshipSatisfaction','StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole','YearsSinceLastPromotion', 'YearsWithCurrManager']


categorical_df = pd.get_dummies(hrdata[cat_variables], drop_first=True)
numerical_df = hrdata[num_variables]


new_df = pd.concat([categorical_df,numerical_df], axis=1)
new_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BusinessTravel_Travel_Frequently</th>
      <th>BusinessTravel_Travel_Rarely</th>
      <th>Department_Research &amp; Development</th>
      <th>Department_Sales</th>
      <th>EducationField_Life Sciences</th>
      <th>EducationField_Marketing</th>
      <th>EducationField_Medical</th>
      <th>EducationField_Other</th>
      <th>EducationField_Technical Degree</th>
      <th>Gender_Male</th>
      <th>...</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>



#### Class Imbalance


```python
new_df.Attrition.value_counts(1)
```




    0    0.838776
    1    0.161224
    Name: Attrition, dtype: float64



***
**Employee Attrition Rate is 16%**
***


```python
plt.figure(figsize=(12,8))
Attrition = new_df.Attrition.value_counts()
sns.barplot(y=Attrition.values, x=Attrition.index, alpha=0.6)
plt.title('Distribution of Employee Attrition')
plt.xlabel('Employee Attrition', fontsize=16)
plt.ylabel('Count', fontsize=16)
```




    Text(0, 0.5, 'Count')




![png](./templates/output_72_1.png)


***
There are many ways of dealing with imbalanced data. We will focus in the following approaches:

1. Oversampling — SMOTE
1. Undersampling — RandomUnderSampler

***

***
#### Let's train a base logistic regression model on the three types of samples to see which yields the best result:

* Orginal Sample
* Upsampling Data
***

***
### Split Dataset into Train and Test set
***


```python
new_df.columns
```




    Index(['BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
           'Department_Research & Development', 'Department_Sales',
           'EducationField_Life Sciences', 'EducationField_Marketing',
           'EducationField_Medical', 'EducationField_Other',
           'EducationField_Technical Degree', 'Gender_Male',
           'JobRole_Human Resources', 'JobRole_Laboratory Technician',
           'JobRole_Manager', 'JobRole_Manufacturing Director',
           'JobRole_Research Director', 'JobRole_Research Scientist',
           'JobRole_Sales Executive', 'JobRole_Sales Representative',
           'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes', 'Age',
           'Attrition', 'DailyRate', 'DistanceFromHome', 'Education',
           'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
           'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
           'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
           'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
           'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
           'YearsSinceLastPromotion', 'YearsWithCurrManager'],
          dtype='object')




```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve

X_cols = ['BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
       'Department_Research & Development', 'Department_Sales',
       'EducationField_Life Sciences', 'EducationField_Marketing',
       'EducationField_Medical', 'EducationField_Other',
       'EducationField_Technical Degree', 'Gender_Male',
       'JobRole_Human Resources', 'JobRole_Laboratory Technician',
       'JobRole_Manager', 'JobRole_Manufacturing Director',
       'JobRole_Research Director', 'JobRole_Research Scientist',
       'JobRole_Sales Executive', 'JobRole_Sales Representative',
       'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes', 'Age',
       'DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
       'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']

y_col = ['Attrition']

# Create the X and y set
X = new_df[X_cols]
y = new_df[y_col]

# Define train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=123, stratify=y)
```


```python
print("The shape of X_train: {}".format(X_train.shape))
print("The shape of y_train: {}".format(y_train.shape))
print("The shape of X_test: {}".format(X_test.shape))
print("The shape of y_test: {}".format(X_train.shape))
```

    The shape of X_train: (1176, 44)
    The shape of y_train: (1176, 1)
    The shape of X_test: (294, 44)
    The shape of y_test: (1176, 44)
    

***
### Resampling
***


```python
!pip install scikit-learn
```

    Requirement already satisfied: scikit-learn in c:\users\teeja\appdata\roaming\python\python37\site-packages (0.23.1)
    Requirement already satisfied: joblib>=0.11 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (0.14.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (2.1.0)
    Requirement already satisfied: scipy>=0.19.1 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.4.1)
    Requirement already satisfied: numpy>=1.13.3 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.18.1)
    


```python
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE 


# Upsample using SMOTE
sm = SMOTE(random_state=12)
x_train_sm, y_train_sm = sm.fit_sample(X_train, np.ravel(y_train, order='C'))



print("Original shape:", X_train.shape, y_train.shape)
print ("SMOTE sample shape:", x_train_sm.shape, y_train_sm.shape)

```

    Original shape: (1176, 44) (1176, 1)
    SMOTE sample shape: (1972, 44) (1972,)
    

#### Base line Logistic Regression for sample Selection


```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Create the Original, Upsampled, and Downsampled training sets
methods_data = {"Original": (X_train, y_train),
                "SMOTE":(x_train_sm,y_train_sm)}

# Loop through each type of training sets and apply 5-Fold CV using Logistic Regression
# By default in cross_val_score StratifiedCV is used
for method in methods_data.keys():
    lr_results = cross_val_score(LogisticRegression(), methods_data[method][0], methods_data[method][1], cv=5, scoring='f1')
    print(f"The best F1 Score for {method} data:")
    print (lr_results.mean())
 
cross_val_score(LogisticRegression(class_weight='balanced'), X_train, y_train, cv=5, scoring='f1').mean()
```

    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    

    The best F1 Score for Original data:
    0.02051282051282051
    

    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    

    The best F1 Score for SMOTE data:
    0.7080648778467554
    

    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    




    0.3835076235251147



* Original Data (class imbalance) : F1 score = 0.5
* Smooth data (balance Class) : F1 score = 0.8


### Modeling

#### Logistic Regression


```python
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

lr = LogisticRegression()

# Fit the model to the Upsampling data
lr = lr.fit(x_train_sm, y_train_sm)

print ("\n\n ---Logistic Regression Model---")
lr_auc = roc_auc_score(y_test, lr.predict(X_test))

print ("Logistic Regression AUC = %2.2f" % lr_auc)

lr2 = lr.fit(x_train_sm, y_train_sm)
print(classification_report(y_test, lr.predict(X_test)))
```

    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    

    
    
     ---Logistic Regression Model---
    Logistic Regression AUC = 0.59
                  precision    recall  f1-score   support
    
               0       0.88      0.64      0.74       247
               1       0.22      0.53      0.31        47
    
        accuracy                           0.63       294
       macro avg       0.55      0.59      0.53       294
    weighted avg       0.77      0.63      0.67       294
    
    

    C:\Users\teeja\AppData\Roaming\Python\Python37\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    

#### Random Forest Classifier


```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
rf = RandomForestClassifier()

rf_result = cross_val_score(rf, x_train_sm, y_train_sm, cv=5, scoring='f1')

rf_result.mean()
```




    0.8829493933992033




```python
from sklearn.metrics import roc_auc_score

rf = rf.fit(x_train_sm, y_train_sm)

print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test)))
```

    
    
     ---Random Forest Model---
    Random Forest AUC = 0.62
                  precision    recall  f1-score   support
    
               0       0.88      0.96      0.92       247
               1       0.59      0.28      0.38        47
    
        accuracy                           0.85       294
       macro avg       0.73      0.62      0.65       294
    weighted avg       0.83      0.85      0.83       294
    
    

#### Gradient Boosting Classifier


```python
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()  

gbc = gbc.fit(x_train_sm,y_train_sm)

gbc
```




    GradientBoostingClassifier()




```python
## 5 fold cross validation
gbc_result = cross_val_score(gbc, x_train_sm, y_train_sm, cv=5, scoring='f1')
gbc_result.mean()
```




    0.8166222018651711




```python
from sklearn.metrics import roc_auc_score

print ("\n\n ---Gradient Boosting Model---")
gbc_auc = roc_auc_score(y_test, gbc.predict(X_test))
print ("Gradient Boosting Classifier AUC = %2.2f" % gbc_auc)
print(classification_report(y_test, gbc.predict(X_test)))
```

    
    
     ---Gradient Boosting Model---
    Gradient Boosting Classifier AUC = 0.65
                  precision    recall  f1-score   support
    
               0       0.89      0.95      0.92       247
               1       0.57      0.36      0.44        47
    
        accuracy                           0.85       294
       macro avg       0.73      0.65      0.68       294
    weighted avg       0.84      0.85      0.84       294
    
    

### ROC Graph


```python
# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test, gbc.predict_proba(X_test)[:,1])


plt.figure()

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % lr_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)

# Plot Decision Tree ROC
plt.plot(gbc_fpr, gbc_tpr, label='Gradient Boosting Classifier (area = %0.2f)' % gbc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()
```


![png](./templates/output_96_0.png)


### Feature Importance


```python
# Get Feature Importances
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances = feature_importances.reset_index()
feature_importances
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>StockOptionLevel</td>
      <td>0.063806</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JobInvolvement</td>
      <td>0.059914</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JobSatisfaction</td>
      <td>0.050951</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MonthlyIncome</td>
      <td>0.046096</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EnvironmentSatisfaction</td>
      <td>0.043649</td>
    </tr>
    <tr>
      <th>5</th>
      <td>YearsAtCompany</td>
      <td>0.040317</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TotalWorkingYears</td>
      <td>0.039259</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MonthlyRate</td>
      <td>0.037338</td>
    </tr>
    <tr>
      <th>8</th>
      <td>YearsInCurrentRole</td>
      <td>0.035976</td>
    </tr>
    <tr>
      <th>9</th>
      <td>JobLevel</td>
      <td>0.035402</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Age</td>
      <td>0.035141</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DailyRate</td>
      <td>0.033795</td>
    </tr>
    <tr>
      <th>12</th>
      <td>YearsWithCurrManager</td>
      <td>0.033074</td>
    </tr>
    <tr>
      <th>13</th>
      <td>HourlyRate</td>
      <td>0.031422</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Department_Research &amp; Development</td>
      <td>0.031261</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DistanceFromHome</td>
      <td>0.030603</td>
    </tr>
    <tr>
      <th>16</th>
      <td>RelationshipSatisfaction</td>
      <td>0.029168</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MaritalStatus_Married</td>
      <td>0.027790</td>
    </tr>
    <tr>
      <th>18</th>
      <td>EducationField_Medical</td>
      <td>0.027121</td>
    </tr>
    <tr>
      <th>19</th>
      <td>TrainingTimesLastYear</td>
      <td>0.025629</td>
    </tr>
    <tr>
      <th>20</th>
      <td>EducationField_Life Sciences</td>
      <td>0.023744</td>
    </tr>
    <tr>
      <th>21</th>
      <td>WorkLifeBalance</td>
      <td>0.021929</td>
    </tr>
    <tr>
      <th>22</th>
      <td>NumCompaniesWorked</td>
      <td>0.021822</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PercentSalaryHike</td>
      <td>0.020257</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Education</td>
      <td>0.019290</td>
    </tr>
    <tr>
      <th>25</th>
      <td>YearsSinceLastPromotion</td>
      <td>0.017221</td>
    </tr>
    <tr>
      <th>26</th>
      <td>BusinessTravel_Travel_Rarely</td>
      <td>0.013708</td>
    </tr>
    <tr>
      <th>27</th>
      <td>JobRole_Research Scientist</td>
      <td>0.013691</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MaritalStatus_Single</td>
      <td>0.012420</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Department_Sales</td>
      <td>0.009848</td>
    </tr>
    <tr>
      <th>30</th>
      <td>OverTime_Yes</td>
      <td>0.009437</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Gender_Male</td>
      <td>0.007689</td>
    </tr>
    <tr>
      <th>32</th>
      <td>JobRole_Laboratory Technician</td>
      <td>0.006214</td>
    </tr>
    <tr>
      <th>33</th>
      <td>JobRole_Sales Executive</td>
      <td>0.005713</td>
    </tr>
    <tr>
      <th>34</th>
      <td>EducationField_Technical Degree</td>
      <td>0.005545</td>
    </tr>
    <tr>
      <th>35</th>
      <td>EducationField_Marketing</td>
      <td>0.005236</td>
    </tr>
    <tr>
      <th>36</th>
      <td>PerformanceRating</td>
      <td>0.004917</td>
    </tr>
    <tr>
      <th>37</th>
      <td>EducationField_Other</td>
      <td>0.004778</td>
    </tr>
    <tr>
      <th>38</th>
      <td>BusinessTravel_Travel_Frequently</td>
      <td>0.004375</td>
    </tr>
    <tr>
      <th>39</th>
      <td>JobRole_Sales Representative</td>
      <td>0.003501</td>
    </tr>
    <tr>
      <th>40</th>
      <td>JobRole_Manufacturing Director</td>
      <td>0.003270</td>
    </tr>
    <tr>
      <th>41</th>
      <td>JobRole_Human Resources</td>
      <td>0.003015</td>
    </tr>
    <tr>
      <th>42</th>
      <td>JobRole_Research Director</td>
      <td>0.002699</td>
    </tr>
    <tr>
      <th>43</th>
      <td>JobRole_Manager</td>
      <td>0.001970</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 9))

# Plot the Feature Importance
sns.set_color_codes("pastel")
sns.barplot(x="importance", y='index', data=feature_importances,
            label="Total", color="b")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x13823625c08>




![png](./templates/output_99_1.png)


#### Apply Random Noise to feature Importance


```python
# Apply Random Noise to data set
X_train_rnoise = pd.DataFrame(X_train)
X_train_rnoise['RANDOM_NOISE'] = np.random.normal(0, 1, X_train_rnoise.shape[0])


# Fit Random Forest to DataSet
rf_random = RandomForestClassifier()
rf_random = rf_random.fit(X_train_rnoise, y_train)

# Get Feature Importances
feature_importances_random = pd.DataFrame(rf_random.feature_importances_, index = X_train_rnoise.columns,columns=['importance']).sort_values('importance', ascending=False)
feature_importances_random = feature_importances_random.reset_index()
feature_importances_random


```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:8: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MonthlyIncome</td>
      <td>0.068249</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TotalWorkingYears</td>
      <td>0.055175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DailyRate</td>
      <td>0.052084</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>0.050917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MonthlyRate</td>
      <td>0.048805</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RANDOM_NOISE</td>
      <td>0.047486</td>
    </tr>
    <tr>
      <th>6</th>
      <td>OverTime_Yes</td>
      <td>0.046186</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HourlyRate</td>
      <td>0.044141</td>
    </tr>
    <tr>
      <th>8</th>
      <td>YearsAtCompany</td>
      <td>0.041089</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DistanceFromHome</td>
      <td>0.039296</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NumCompaniesWorked</td>
      <td>0.035111</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PercentSalaryHike</td>
      <td>0.033514</td>
    </tr>
    <tr>
      <th>12</th>
      <td>StockOptionLevel</td>
      <td>0.030785</td>
    </tr>
    <tr>
      <th>13</th>
      <td>YearsWithCurrManager</td>
      <td>0.030570</td>
    </tr>
    <tr>
      <th>14</th>
      <td>EnvironmentSatisfaction</td>
      <td>0.027433</td>
    </tr>
    <tr>
      <th>15</th>
      <td>JobSatisfaction</td>
      <td>0.026685</td>
    </tr>
    <tr>
      <th>16</th>
      <td>JobInvolvement</td>
      <td>0.026648</td>
    </tr>
    <tr>
      <th>17</th>
      <td>YearsInCurrentRole</td>
      <td>0.025143</td>
    </tr>
    <tr>
      <th>18</th>
      <td>YearsSinceLastPromotion</td>
      <td>0.024680</td>
    </tr>
    <tr>
      <th>19</th>
      <td>WorkLifeBalance</td>
      <td>0.024249</td>
    </tr>
    <tr>
      <th>20</th>
      <td>TrainingTimesLastYear</td>
      <td>0.023505</td>
    </tr>
    <tr>
      <th>21</th>
      <td>RelationshipSatisfaction</td>
      <td>0.021855</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Education</td>
      <td>0.019831</td>
    </tr>
    <tr>
      <th>23</th>
      <td>JobLevel</td>
      <td>0.018186</td>
    </tr>
    <tr>
      <th>24</th>
      <td>MaritalStatus_Single</td>
      <td>0.013361</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BusinessTravel_Travel_Frequently</td>
      <td>0.011904</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Gender_Male</td>
      <td>0.009555</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Department_Research &amp; Development</td>
      <td>0.008962</td>
    </tr>
    <tr>
      <th>28</th>
      <td>JobRole_Sales Representative</td>
      <td>0.008687</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Department_Sales</td>
      <td>0.008159</td>
    </tr>
    <tr>
      <th>30</th>
      <td>EducationField_Life Sciences</td>
      <td>0.007963</td>
    </tr>
    <tr>
      <th>31</th>
      <td>MaritalStatus_Married</td>
      <td>0.007804</td>
    </tr>
    <tr>
      <th>32</th>
      <td>EducationField_Technical Degree</td>
      <td>0.007778</td>
    </tr>
    <tr>
      <th>33</th>
      <td>JobRole_Laboratory Technician</td>
      <td>0.007752</td>
    </tr>
    <tr>
      <th>34</th>
      <td>JobRole_Research Scientist</td>
      <td>0.007234</td>
    </tr>
    <tr>
      <th>35</th>
      <td>JobRole_Sales Executive</td>
      <td>0.006584</td>
    </tr>
    <tr>
      <th>36</th>
      <td>BusinessTravel_Travel_Rarely</td>
      <td>0.006408</td>
    </tr>
    <tr>
      <th>37</th>
      <td>EducationField_Medical</td>
      <td>0.006290</td>
    </tr>
    <tr>
      <th>38</th>
      <td>EducationField_Marketing</td>
      <td>0.004717</td>
    </tr>
    <tr>
      <th>39</th>
      <td>EducationField_Other</td>
      <td>0.003678</td>
    </tr>
    <tr>
      <th>40</th>
      <td>JobRole_Human Resources</td>
      <td>0.003255</td>
    </tr>
    <tr>
      <th>41</th>
      <td>PerformanceRating</td>
      <td>0.003233</td>
    </tr>
    <tr>
      <th>42</th>
      <td>JobRole_Manufacturing Director</td>
      <td>0.002827</td>
    </tr>
    <tr>
      <th>43</th>
      <td>JobRole_Manager</td>
      <td>0.001158</td>
    </tr>
    <tr>
      <th>44</th>
      <td>JobRole_Research Director</td>
      <td>0.001073</td>
    </tr>
  </tbody>
</table>
</div>



#### Viz Feature Importance


```python

# Create Seaborn PLot
sns.set(style="whitegrid")
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 9))


clrs = ['red' if (x == 9 ) else 'grey' for x in feature_importances_random.index.values ]

# Plot the Feature Importance
sns.barplot(x="importance", y='index', data=feature_importances_random,
            label="Total",  palette=clrs)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x138235a2a88>




![png](./templates/output_103_1.png)


### Random Forest Using Feature Importance


```python
col_imp = ['Attrition','Age','OverTime','HourlyRate','DailyRate','MonthlyIncome','TotalWorkingYears','YearsAtCompany','NumCompaniesWorked','DistanceFromHome']

```


```python
x_train_sm.shape
```




    (1972, 44)



#### Select the feature importance columns


```python
hrdata_imp = hrdata[col_imp ]
hrdata_imp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   Attrition           1470 non-null   int64 
     1   Age                 1470 non-null   int64 
     2   OverTime            1470 non-null   object
     3   HourlyRate          1470 non-null   int64 
     4   DailyRate           1470 non-null   int64 
     5   MonthlyIncome       1470 non-null   int64 
     6   TotalWorkingYears   1470 non-null   int64 
     7   YearsAtCompany      1470 non-null   int64 
     8   NumCompaniesWorked  1470 non-null   int64 
     9   DistanceFromHome    1470 non-null   int64 
    dtypes: int64(9), object(1)
    memory usage: 115.0+ KB
    

#### Correlation Matrix of the selected Columns


```python
## corelation matrix
corr = hrdata_imp.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels= corr.columns.values)
plt.title("Heatmap of Correlation Matrix of the IBM HR datasets")
corr

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attrition</th>
      <th>Age</th>
      <th>HourlyRate</th>
      <th>DailyRate</th>
      <th>MonthlyIncome</th>
      <th>TotalWorkingYears</th>
      <th>YearsAtCompany</th>
      <th>NumCompaniesWorked</th>
      <th>DistanceFromHome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>1.000000</td>
      <td>-0.159205</td>
      <td>-0.006846</td>
      <td>-0.056652</td>
      <td>-0.159840</td>
      <td>-0.171063</td>
      <td>-0.134392</td>
      <td>0.043494</td>
      <td>0.077924</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.159205</td>
      <td>1.000000</td>
      <td>0.024287</td>
      <td>0.010661</td>
      <td>0.497855</td>
      <td>0.680381</td>
      <td>0.311309</td>
      <td>0.299635</td>
      <td>-0.001686</td>
    </tr>
    <tr>
      <th>HourlyRate</th>
      <td>-0.006846</td>
      <td>0.024287</td>
      <td>1.000000</td>
      <td>0.023381</td>
      <td>-0.015794</td>
      <td>-0.002334</td>
      <td>-0.019582</td>
      <td>0.022157</td>
      <td>0.031131</td>
    </tr>
    <tr>
      <th>DailyRate</th>
      <td>-0.056652</td>
      <td>0.010661</td>
      <td>0.023381</td>
      <td>1.000000</td>
      <td>0.007707</td>
      <td>0.014515</td>
      <td>-0.034055</td>
      <td>0.038153</td>
      <td>-0.004985</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>-0.159840</td>
      <td>0.497855</td>
      <td>-0.015794</td>
      <td>0.007707</td>
      <td>1.000000</td>
      <td>0.772893</td>
      <td>0.514285</td>
      <td>0.149515</td>
      <td>-0.017014</td>
    </tr>
    <tr>
      <th>TotalWorkingYears</th>
      <td>-0.171063</td>
      <td>0.680381</td>
      <td>-0.002334</td>
      <td>0.014515</td>
      <td>0.772893</td>
      <td>1.000000</td>
      <td>0.628133</td>
      <td>0.237639</td>
      <td>0.004628</td>
    </tr>
    <tr>
      <th>YearsAtCompany</th>
      <td>-0.134392</td>
      <td>0.311309</td>
      <td>-0.019582</td>
      <td>-0.034055</td>
      <td>0.514285</td>
      <td>0.628133</td>
      <td>1.000000</td>
      <td>-0.118421</td>
      <td>0.009508</td>
    </tr>
    <tr>
      <th>NumCompaniesWorked</th>
      <td>0.043494</td>
      <td>0.299635</td>
      <td>0.022157</td>
      <td>0.038153</td>
      <td>0.149515</td>
      <td>0.237639</td>
      <td>-0.118421</td>
      <td>1.000000</td>
      <td>-0.029251</td>
    </tr>
    <tr>
      <th>DistanceFromHome</th>
      <td>0.077924</td>
      <td>-0.001686</td>
      <td>0.031131</td>
      <td>-0.004985</td>
      <td>-0.017014</td>
      <td>0.004628</td>
      <td>0.009508</td>
      <td>-0.029251</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png](./templates/output_110_1.png)


#### Data Pre-processing


```python
hrdata_imp.dtypes
```




    Attrition              int64
    Age                    int64
    OverTime              object
    HourlyRate             int64
    DailyRate              int64
    MonthlyIncome          int64
    TotalWorkingYears      int64
    YearsAtCompany         int64
    NumCompaniesWorked     int64
    DistanceFromHome       int64
    dtype: object




```python
cat_df = pd.get_dummies(hrdata_imp[['OverTime']], drop_first=True)
num_df = hrdata_imp[['Attrition','Age','HourlyRate','DailyRate','MonthlyIncome','TotalWorkingYears','YearsAtCompany','NumCompaniesWorked','DistanceFromHome']]


new_df_imp = pd.concat([num_df,cat_df], axis=1)
new_df_imp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attrition</th>
      <th>Age</th>
      <th>HourlyRate</th>
      <th>DailyRate</th>
      <th>MonthlyIncome</th>
      <th>TotalWorkingYears</th>
      <th>YearsAtCompany</th>
      <th>NumCompaniesWorked</th>
      <th>DistanceFromHome</th>
      <th>OverTime_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>41</td>
      <td>94</td>
      <td>1102</td>
      <td>5993</td>
      <td>8</td>
      <td>6</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>49</td>
      <td>61</td>
      <td>279</td>
      <td>5130</td>
      <td>10</td>
      <td>10</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>37</td>
      <td>92</td>
      <td>1373</td>
      <td>2090</td>
      <td>7</td>
      <td>0</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>33</td>
      <td>56</td>
      <td>1392</td>
      <td>2909</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>27</td>
      <td>40</td>
      <td>591</td>
      <td>3468</td>
      <td>6</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Split Train/Test


```python
new_df_imp.columns
```




    Index(['Attrition', 'Age', 'HourlyRate', 'DailyRate', 'MonthlyIncome',
           'TotalWorkingYears', 'YearsAtCompany', 'NumCompaniesWorked',
           'DistanceFromHome', 'OverTime_Yes'],
          dtype='object')




```python
x_cols = [ 'Age', 'HourlyRate', 'DailyRate', 'MonthlyIncome','TotalWorkingYears', 'YearsAtCompany', 'NumCompaniesWorked','DistanceFromHome', 'OverTime_Yes']
y_col = ['Attrition']

# Create the X and y set
X = new_df_imp[x_cols]
y = new_df_imp[y_col]

# Define train and test
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X,y,test_size=0.20, random_state=123, stratify=y)
```


```python
print("X_train_imp shape : {}".format(X_train_imp.shape))
print("y_train_imp shape : {}".format(y_train_imp.shape))
print("X_test_imp shape : {}".format(X_test_imp.shape))
print("y_test_imp shape : {}".format(y_test_imp.shape))
```

    X_train_imp shape : (1176, 9)
    y_train_imp shape : (1176, 1)
    X_test_imp shape : (294, 9)
    y_test_imp shape : (294, 1)
    

#### Resampling


```python
# Upsample using SMOTE
sm = SMOTE(random_state=12)
X_train_imp_sm, y_train_imp_sm = sm.fit_sample(X_train_imp,np.ravel(y_train_imp, order='C'))



print("Original shape:", X_train_imp.shape, y_train_imp.shape)
print ("SMOTE sample shape:", X_train_imp_sm.shape, y_train_imp_sm.shape)
```

    Original shape: (1176, 9) (1176, 1)
    SMOTE sample shape: (1972, 9) (1972,)
    

***
### Random Forest
***


```python
# Random Forest Model
model = RandomForestClassifier()

rf_result = cross_val_score(model, X_train_imp_sm, y_train_imp_sm, cv=5, scoring='f1')

rf_result.mean()
```




    0.8607292399007177




```python
X_train_imp_sm_df = pd.DataFrame(X_train_imp_sm)


# Fit Random Forest to DataSet
model = RandomForestClassifier()
model_rf = model.fit(X_train_imp_sm_df, y_train_imp_sm)


# Get Feature Importances
feature = pd.DataFrame(model_rf.feature_importances_, index = X_train_imp_sm_df.columns,columns=['importance']).sort_values('importance', ascending=False)
feature = feature.reset_index()
feature

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MonthlyIncome</td>
      <td>0.181915</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DailyRate</td>
      <td>0.149468</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YearsAtCompany</td>
      <td>0.122780</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>0.121806</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HourlyRate</td>
      <td>0.120830</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TotalWorkingYears</td>
      <td>0.106912</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DistanceFromHome</td>
      <td>0.100162</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NumCompaniesWorked</td>
      <td>0.069163</td>
    </tr>
    <tr>
      <th>8</th>
      <td>OverTime_Yes</td>
      <td>0.026964</td>
    </tr>
  </tbody>
</table>
</div>



***
#### Test the model
***


```python
prediction = model_rf.predict(X_test_imp)
prediction[:10]
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64)




```python
print ("\n\n ---Random Forest Model---")
rfm_auc = roc_auc_score(y_test_imp,prediction )

print ("Random Forest Classifier AUC = %2.2f" % rfm_auc)
print(classification_report(y_test_imp, prediction))
```

    
    
     ---Random Forest Model---
    Random Forest Classifier AUC = 0.65
                  precision    recall  f1-score   support
    
               0       0.89      0.86      0.88       247
               1       0.38      0.45      0.41        47
    
        accuracy                           0.80       294
       macro avg       0.64      0.65      0.64       294
    weighted avg       0.81      0.80      0.80       294
    
    


```python
model_rf.predict_proba(X_test_imp)[:30]
```




    array([[0.56, 0.44],
           [0.59, 0.41],
           [0.88, 0.12],
           [0.68, 0.32],
           [0.74, 0.26],
           [0.76, 0.24],
           [0.54, 0.46],
           [0.59, 0.41],
           [0.64, 0.36],
           [0.94, 0.06],
           [0.87, 0.13],
           [0.25, 0.75],
           [0.55, 0.45],
           [0.47, 0.53],
           [0.45, 0.55],
           [0.83, 0.17],
           [0.75, 0.25],
           [0.6 , 0.4 ],
           [0.54, 0.46],
           [0.91, 0.09],
           [0.12, 0.88],
           [0.61, 0.39],
           [0.62, 0.38],
           [0.88, 0.12],
           [0.29, 0.71],
           [0.7 , 0.3 ],
           [0.32, 0.68],
           [0.81, 0.19],
           [0.85, 0.15],
           [0.85, 0.15]])



#### save the model


```python
import pickle

# save the model to disk
filename = 'random_forest_hr_model.sav'
pickle.dump(model_rf, open(filename, 'wb'))


# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
```


```python

def prep_data(df):
    '''
    :Assumption: Asuuming the dataframe contains the required columns
    : required columns : ['OverTime','Age','HourlyRate','DailyRate','MonthlyIncome','TotalWorkingYears','YearsAtCompany','NumCompaniesWorked','DistanceFromHome']
    :input: pandas dataframe
    :output: pre-processed dataframe  with selected columns
    '''
    try :
        cat_df = pd.get_dummies(df[['OverTime']], drop_first=True)
        num_df = df[['Age','HourlyRate','DailyRate','MonthlyIncome','TotalWorkingYears','YearsAtCompany','NumCompaniesWorked','DistanceFromHome']]
        new_df = pd.concat([num_df,cat_df], axis=1)
        return new_df
    except Exception as e:
        cols = ['OverTime','Age','HourlyRate','DailyRate','MonthlyIncome','TotalWorkingYears','YearsAtCompany','NumCompaniesWorked','DistanceFromHome']
        print("\nn___Required Columns Not in Dataframe__\n\n")
        print(cols)
        print(str(e))
        print("\n\n__ The function returns None__")
        return None
```


```python
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

new_prediction = loaded_model.predict(prep_data(hrdata))
```


```python
print ("\n\n ---Random Forest Model---")
rfm_auc = roc_auc_score(hrdata['Attrition'],new_prediction )

print ("Random Forest Classifier AUC = %2.2f" % rfm_auc)
print(classification_report(hrdata['Attrition'], new_prediction))
```

    
    
     ---Random Forest Model---
    Random Forest Classifier AUC = 0.93
                  precision    recall  f1-score   support
    
               0       0.98      0.97      0.98      1233
               1       0.86      0.89      0.88       237
    
        accuracy                           0.96      1470
       macro avg       0.92      0.93      0.93      1470
    weighted avg       0.96      0.96      0.96      1470
    
    


```python
temp = hrdata[['HourlyRate','DailyRate','MonthlyIncome','TotalWorkingYears']]

x_temp = prep_data(temp)
```

    
    n___Required Columns Not in Dataframe__
    
    
    ['OverTime', 'Age', 'HourlyRate', 'DailyRate', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'NumCompaniesWorked', 'DistanceFromHome']
    "None of [Index(['OverTime'], dtype='object')] are in the [columns]"
    
    
    __ The function returns None__
    


```python
 prob = loaded_model.predict_proba(prep_data(hrdata))
 prob[:10]
    
    
```




    array([[0.18, 0.82],
           [0.94, 0.06],
           [0.14, 0.86],
           [0.85, 0.15],
           [0.5 , 0.5 ],
           [0.83, 0.17],
           [0.9 , 0.1 ],
           [0.73, 0.27],
           [0.89, 0.11],
           [0.96, 0.04]])




```python
len(prob)
```




    1470




```python
for el in prob[:10]:
    print("The probability of Employee Attrition is : {} %".format(el[0] * 100) )
```

    The probability of Employee Attrition is : 18.0 %
    The probability of Employee Attrition is : 94.0 %
    The probability of Employee Attrition is : 14.000000000000002 %
    The probability of Employee Attrition is : 85.0 %
    The probability of Employee Attrition is : 50.0 %
    The probability of Employee Attrition is : 83.0 %
    The probability of Employee Attrition is : 90.0 %
    The probability of Employee Attrition is : 73.0 %
    The probability of Employee Attrition is : 89.0 %
    The probability of Employee Attrition is : 96.0 %
    


```python

```
