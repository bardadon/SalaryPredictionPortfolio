```python
__author__ = "Bar Dadon"
__email__ = "bdadon50@gmail.com"
```

# <ins> Understanding and Predicting Employees Salary

## Table of Contents

### The Problem 
- [Why Do We Need to Predict employees salaries?](#problem)

### Data Quality Check And Cleaning 
- [Simple Inspection of Data](#dataInspect)
- [Summery Statistics](#Summery)

### Exploratory data analysis
- [Salary Inspection](#salaryInspect)
- [Outliers Detection](#outliersDetection)
- [Features Distribution](#featuresDistribution)
- [Features Analysis](#FeaturesAnalysis)
- [Correlation Analysis](#CorrelationAnalysis)
- [Baseline Model](#BaselineModel)

### Pre-Processing
- [Categorical Data](#CategoricalData)
- [Creating Train and Target Data Sets](#TrainAndTarget)

### Develop
- [Creating Models](#CreatingModels)
- [Model Selection](#ModelSelection)

### Deploy
- [Pre:Processing Test data](#TestData)
- [Feature Importance](#FeatureImportance)

<a id= 'problem' ></a>
# <ins> The Problem 

 - __The Goal of This Project__ Is to Create a Model that Hr Teams Can Use to __Predict Their Employees Salaries and Accuratley Post Job Offers__ with the Right Salary that Will Appeal to the Right People.
by Creating a Fast and Accurate Model that Predicts Salaries, __Companies Can Invest Their Hr Resources and Personnel__ Into Different Initiatives Like Properly __Filter Through Candidates__ or __Investing in Reducing Employees Turnover__ and So On.
It Can Reduce Costs for The Company by Eliminating Cases Where the Employees Were Offered Salaries that Weren't Compatible with Their Value to The Company
and Thus Were Over Paid, While Also Reduce Losing Applicants Because of Under Offers of Salaries.

- <ins> __The Data at Our Disposal Consists of 1m Samples of Employees that Are Defined by The Features:__</ins>
    1. Job Id	
    - Company Id	
    - Job Type	
    - Degree	
    - Major	
    - Industry	
    - Years Of Experience	
    - Miles from Metropolis


- The Metric that Will Be Used in This Project Is MSE(Mean Squared Error)



# <ins> Import Packages


```python
#import your libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing , svm
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
%matplotlib inline
```

# <ins> General functions


```python
def load_file(file):
    '''Loads a csv file'''
    return pd.read_csv(file)

def consolidate_data(df1, df2, key=None, left_index=False, right_index=False):
    '''Merge two Data Frames, return only records that appear in both data sets'''
    return pd.merge(left = df1, right = df2, how = 'inner',on = key, left_index=False, right_index=False)

def describe_dataframe(df=pd.DataFrame()):
    """This function generates descriptive stats of a dataframe
    Args:
        df (dataframe): the dataframe to be analyzed
    Returns:
        None

    """
    print("\n\n")
    print("*"*30)
    print("About the Data")
    print("*"*30)
    
    print("Number of rows::",df.shape[0])
    print("Number of columns::",df.shape[1])
    print("\n")
    
    print("Column Names::",df.columns.values.tolist())
    print("\n")
    
    print("Column Data Types::\n",df.dtypes)
    print("\n")
    
    print("Columns with Missing Values::",df.columns[df.isnull().any()].tolist())
    print("\n")
    
    print("Number of rows with Missing Values::",len(pd.isnull(df).any(1).to_numpy().nonzero()[0].tolist()))
    print("\n")
    
    print("Sample Indices with missing data::",pd.isnull(df).any(1).to_numpy().nonzero()[0].tolist()[0:5])
    print("\n")
    
    print("General Stats::")
    print(df.info())
    print("\n")
    
    print('Summery Stats(Numerical): ')
    print(df.describe())
    print('\n')
    
    
    print('Summery Stats(Objects): ')
    print(df.astype('object').describe().transpose())
    print('\n')
    
    print("Dataframe Sample Rows::")
    display(df.head(5))
```

# <ins> Data Quality Check and Cleaning

<a id= dataInspect ></a>
## Read the Data


```python
#load the data into a Pandas dataframe
train_features = load_file('C:/Users/user/Desktop/Data Science/DSDJ/Module 4 - Portfolio/train_features.csv')
train_salaries = load_file('C:/Users/user/Desktop/Data Science/DSDJ/Module 4 - Portfolio/train_salaries.csv')
test_features = load_file('C:/Users/user/Desktop/Data Science/DSDJ/Module 4 - Portfolio/test_features.csv')
```


```python
train_features.head(10)
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
      <th>jobId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JOB1362684407687</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>MATH</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JOB1362684407688</td>
      <td>CEO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>WEB</td>
      <td>3</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JOB1362684407689</td>
      <td>VICE_PRESIDENT</td>
      <td>DOCTORAL</td>
      <td>PHYSICS</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JOB1362684407690</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>CHEMISTRY</td>
      <td>AUTO</td>
      <td>8</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOB1362684407691</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>PHYSICS</td>
      <td>FINANCE</td>
      <td>8</td>
      <td>16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>JOB1362684407692</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>COMPSCI</td>
      <td>FINANCE</td>
      <td>2</td>
      <td>31</td>
    </tr>
    <tr>
      <th>6</th>
      <td>JOB1362684407693</td>
      <td>CFO</td>
      <td>NONE</td>
      <td>NONE</td>
      <td>HEALTH</td>
      <td>23</td>
      <td>24</td>
    </tr>
    <tr>
      <th>7</th>
      <td>JOB1362684407694</td>
      <td>JUNIOR</td>
      <td>BACHELORS</td>
      <td>CHEMISTRY</td>
      <td>EDUCATION</td>
      <td>9</td>
      <td>70</td>
    </tr>
    <tr>
      <th>8</th>
      <td>JOB1362684407695</td>
      <td>JANITOR</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>EDUCATION</td>
      <td>1</td>
      <td>54</td>
    </tr>
    <tr>
      <th>9</th>
      <td>JOB1362684407696</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>CHEMISTRY</td>
      <td>AUTO</td>
      <td>17</td>
      <td>68</td>
    </tr>
  </tbody>
</table>
</div>



- <span style = 'color:red' >All the Features Seem Relevant to The Target Variable, __Except Company Id Which Adds No Value__ ----> Dropping Company Id Column </span>


```python
train_features = train_features.drop('companyId',axis=1)
train_features
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
      <th>jobId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JOB1362684407687</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>MATH</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JOB1362684407688</td>
      <td>CEO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>WEB</td>
      <td>3</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JOB1362684407689</td>
      <td>VICE_PRESIDENT</td>
      <td>DOCTORAL</td>
      <td>PHYSICS</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JOB1362684407690</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>CHEMISTRY</td>
      <td>AUTO</td>
      <td>8</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOB1362684407691</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>PHYSICS</td>
      <td>FINANCE</td>
      <td>8</td>
      <td>16</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>999995</th>
      <td>JOB1362685407682</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>CHEMISTRY</td>
      <td>HEALTH</td>
      <td>19</td>
      <td>94</td>
    </tr>
    <tr>
      <th>999996</th>
      <td>JOB1362685407683</td>
      <td>CTO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>FINANCE</td>
      <td>12</td>
      <td>35</td>
    </tr>
    <tr>
      <th>999997</th>
      <td>JOB1362685407684</td>
      <td>JUNIOR</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>EDUCATION</td>
      <td>16</td>
      <td>81</td>
    </tr>
    <tr>
      <th>999998</th>
      <td>JOB1362685407685</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>NONE</td>
      <td>HEALTH</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>999999</th>
      <td>JOB1362685407686</td>
      <td>JUNIOR</td>
      <td>BACHELORS</td>
      <td>NONE</td>
      <td>EDUCATION</td>
      <td>20</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
<p>1000000 rows × 7 columns</p>
</div>



- <span style = 'color:red' >~~All the Features Seem Relevant to The Target Variable, Except Company Id Which Adds No Value ----> Droping Company Id Column ~~</span>


```python
train_salaries.head(10)
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
      <th>jobId</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JOB1362684407687</td>
      <td>130</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JOB1362684407688</td>
      <td>101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JOB1362684407689</td>
      <td>137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JOB1362684407690</td>
      <td>142</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOB1362684407691</td>
      <td>163</td>
    </tr>
    <tr>
      <th>5</th>
      <td>JOB1362684407692</td>
      <td>113</td>
    </tr>
    <tr>
      <th>6</th>
      <td>JOB1362684407693</td>
      <td>178</td>
    </tr>
    <tr>
      <th>7</th>
      <td>JOB1362684407694</td>
      <td>73</td>
    </tr>
    <tr>
      <th>8</th>
      <td>JOB1362684407695</td>
      <td>31</td>
    </tr>
    <tr>
      <th>9</th>
      <td>JOB1362684407696</td>
      <td>104</td>
    </tr>
  </tbody>
</table>
</div>






```python
train_features.isnull().any()
train_salaries.isnull().any()
```




    jobId     False
    salary    False
    dtype: bool



- <span style = 'color:green' >Train Data Doesnt Contain Any Null Values</span>

##  Merging the Features with The Target Variable(Salary)


```python
train = consolidate_data(train_features,train_salaries)
```

<a id = Summery ></a>
## Summery Statistics


```python
describe_dataframe(train)
```

    
    
    
    ******************************
    About the Data
    ******************************
    Number of rows:: 1000000
    Number of columns:: 8
    
    
    Column Names:: ['jobId', 'jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis', 'salary']
    
    
    Column Data Types::
     jobId                  object
    jobType                object
    degree                 object
    major                  object
    industry               object
    yearsExperience         int64
    milesFromMetropolis     int64
    salary                  int64
    dtype: object
    
    
    Columns with Missing Values:: []
    
    
    Number of rows with Missing Values:: 0
    
    
    Sample Indices with missing data:: []
    
    
    General Stats::
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1000000 entries, 0 to 999999
    Data columns (total 8 columns):
     #   Column               Non-Null Count    Dtype 
    ---  ------               --------------    ----- 
     0   jobId                1000000 non-null  object
     1   jobType              1000000 non-null  object
     2   degree               1000000 non-null  object
     3   major                1000000 non-null  object
     4   industry             1000000 non-null  object
     5   yearsExperience      1000000 non-null  int64 
     6   milesFromMetropolis  1000000 non-null  int64 
     7   salary               1000000 non-null  int64 
    dtypes: int64(3), object(5)
    memory usage: 68.7+ MB
    None
    
    
    Summery Stats(Numerical): 
           yearsExperience  milesFromMetropolis          salary
    count   1000000.000000       1000000.000000  1000000.000000
    mean         11.992386            49.529260      116.061818
    std           7.212391            28.877733       38.717936
    min           0.000000             0.000000        0.000000
    25%           6.000000            25.000000       88.000000
    50%          12.000000            50.000000      114.000000
    75%          18.000000            75.000000      141.000000
    max          24.000000            99.000000      301.000000
    
    
    Summery Stats(Objects): 
                           count   unique               top    freq
    jobId                1000000  1000000  JOB1362685065451       1
    jobType              1000000        8            SENIOR  125886
    degree               1000000        5       HIGH_SCHOOL  236976
    major                1000000        9              NONE  532355
    industry             1000000        7               WEB  143206
    yearsExperience      1000000       25                15   40312
    milesFromMetropolis  1000000      100                99   10180
    salary               1000000      280               108   10467
    
    
    Dataframe Sample Rows::
    


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
      <th>jobId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JOB1362684407687</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>MATH</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>83</td>
      <td>130</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JOB1362684407688</td>
      <td>CEO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>WEB</td>
      <td>3</td>
      <td>73</td>
      <td>101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JOB1362684407689</td>
      <td>VICE_PRESIDENT</td>
      <td>DOCTORAL</td>
      <td>PHYSICS</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>38</td>
      <td>137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JOB1362684407690</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>CHEMISTRY</td>
      <td>AUTO</td>
      <td>8</td>
      <td>17</td>
      <td>142</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOB1362684407691</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>PHYSICS</td>
      <td>FINANCE</td>
      <td>8</td>
      <td>16</td>
      <td>163</td>
    </tr>
  </tbody>
</table>
</div>


- <span style = 'color:red' >Train Data Contain __Salaries that Are Equal to Zero__ ----> Further Inspection Required</span>


```python
zero_salary_employees = train[train['salary'] == 0]
zero_salary_employees
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
      <th>jobId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30559</th>
      <td>JOB1362684438246</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>MATH</td>
      <td>AUTO</td>
      <td>11</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>495984</th>
      <td>JOB1362684903671</td>
      <td>JUNIOR</td>
      <td>NONE</td>
      <td>NONE</td>
      <td>OIL</td>
      <td>1</td>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>652076</th>
      <td>JOB1362685059763</td>
      <td>CTO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>AUTO</td>
      <td>6</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>816129</th>
      <td>JOB1362685223816</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>ENGINEERING</td>
      <td>FINANCE</td>
      <td>18</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>828156</th>
      <td>JOB1362685235843</td>
      <td>VICE_PRESIDENT</td>
      <td>MASTERS</td>
      <td>ENGINEERING</td>
      <td>WEB</td>
      <td>3</td>
      <td>29</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



- Some Jobs Belong to Educated, High Ranking Employees with A Lot of Years of Experience.
There's Reason to Believe that __The Salaries Were Generated Due to Corrupted Data__ 
- In Addition, There Are only 5 Employees with 0 Salary, Dropping Them Won't Affect the Data Set


```python
train = train[train['salary'] > 0]
```

- <span style = 'color:red' >~~Train Data Contain __Salaries that Are Equal to Zero__ ----> Further Inspection Required~~</span>


```python
train['jobId'].duplicated().sum()
```




    0



- <span style = 'color:green' >Train Data Doesn't Contain Any Duplicated Values</span>

# <ins> Exploratory Data Analysis

<a id = salaryInspect ></a>
##  EDA 1 - Target Variable(Salary) Inspection


```python
train.salary.describe()
```




    count    999995.000000
    mean        116.062398
    std          38.717163
    min          17.000000
    25%          88.000000
    50%         114.000000
    75%         141.000000
    max         301.000000
    Name: salary, dtype: float64



- Salaries Ranges Between 17 to 301
- The Mean Is 116 and The Median Is 114,meaning the Data Most Likely Isn't Skewed
- Most Salaries Are Between 88 to 141 
- <span style = 'color:red' >Considering the Overall Range It Means that __There Are a Few Extreme Outliers__ ----> Further Inspection Required </span>
    


```python
salary_dist = train['salary'].plot.hist(bins=30)
plt.xlabel('Salary')
plt.ylabel('Count')
plt.title('Salary Distribution')
plt.legend(loc = 'best')
plt.show()
```


![png](output_41_0.png)


- <span style = 'color:green'> The Data Looks Symetrical, Visually There's No Reason to Manipulate The Data </span>

<a id = outliersDetection></a>
### IQR rule to identify potential outliers


```python
stat = train.salary.describe()
IQR = stat['75%'] - stat['25%']
upper = stat['75%'] + 1.5 * IQR
lower = stat['25%'] - 1.5 * IQR
if lower < 0:
    lower = 0
    
sns.boxplot(train.salary)
print('The Upper and Lower Bounds for Suspected Outliers Are {} and {}.'.format(upper, lower))
```

    The Upper and Lower Bounds for Suspected Outliers Are 220.5 and 8.5.
    


![png](output_44_1.png)


<ins> __Checking for Corrupted Data for The Outliers__
- Checking for Any Logical Errors About High Payed Employees(salary Above 220.5)


```python
high_salary_outliers = train[train['salary'] > 220.5]
high_salary_outliers
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
      <th>jobId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>266</th>
      <td>JOB1362684407953</td>
      <td>CEO</td>
      <td>MASTERS</td>
      <td>BIOLOGY</td>
      <td>OIL</td>
      <td>23</td>
      <td>60</td>
      <td>223</td>
    </tr>
    <tr>
      <th>362</th>
      <td>JOB1362684408049</td>
      <td>CTO</td>
      <td>MASTERS</td>
      <td>NONE</td>
      <td>HEALTH</td>
      <td>24</td>
      <td>3</td>
      <td>223</td>
    </tr>
    <tr>
      <th>560</th>
      <td>JOB1362684408247</td>
      <td>CEO</td>
      <td>MASTERS</td>
      <td>BIOLOGY</td>
      <td>WEB</td>
      <td>22</td>
      <td>7</td>
      <td>248</td>
    </tr>
    <tr>
      <th>670</th>
      <td>JOB1362684408357</td>
      <td>CEO</td>
      <td>MASTERS</td>
      <td>MATH</td>
      <td>AUTO</td>
      <td>23</td>
      <td>9</td>
      <td>240</td>
    </tr>
    <tr>
      <th>719</th>
      <td>JOB1362684408406</td>
      <td>VICE_PRESIDENT</td>
      <td>DOCTORAL</td>
      <td>BIOLOGY</td>
      <td>OIL</td>
      <td>21</td>
      <td>14</td>
      <td>225</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>998516</th>
      <td>JOB1362685406203</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>ENGINEERING</td>
      <td>WEB</td>
      <td>14</td>
      <td>46</td>
      <td>227</td>
    </tr>
    <tr>
      <th>999249</th>
      <td>JOB1362685406936</td>
      <td>CEO</td>
      <td>NONE</td>
      <td>NONE</td>
      <td>OIL</td>
      <td>17</td>
      <td>10</td>
      <td>223</td>
    </tr>
    <tr>
      <th>999280</th>
      <td>JOB1362685406967</td>
      <td>CFO</td>
      <td>BACHELORS</td>
      <td>BUSINESS</td>
      <td>SERVICE</td>
      <td>21</td>
      <td>0</td>
      <td>228</td>
    </tr>
    <tr>
      <th>999670</th>
      <td>JOB1362685407357</td>
      <td>CEO</td>
      <td>DOCTORAL</td>
      <td>LITERATURE</td>
      <td>SERVICE</td>
      <td>24</td>
      <td>14</td>
      <td>233</td>
    </tr>
    <tr>
      <th>999893</th>
      <td>JOB1362685407580</td>
      <td>CEO</td>
      <td>DOCTORAL</td>
      <td>ENGINEERING</td>
      <td>FINANCE</td>
      <td>17</td>
      <td>33</td>
      <td>237</td>
    </tr>
  </tbody>
</table>
<p>7117 rows × 8 columns</p>
</div>



- There Are About 7000 Employees with High Paying Jobs(Salary Above 220.5)


```python
high_salary_outliers = high_salary_outliers.set_index('salary').sort_values('salary',ascending = False)
```


```python
high_salary_outliers.groupby('jobType').describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">yearsExperience</th>
      <th colspan="8" halign="left">milesFromMetropolis</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>jobType</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CEO</th>
      <td>3227.0</td>
      <td>19.475984</td>
      <td>4.171720</td>
      <td>0.0</td>
      <td>17.00</td>
      <td>21.0</td>
      <td>23.0</td>
      <td>24.0</td>
      <td>3227.0</td>
      <td>23.017354</td>
      <td>19.403017</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>18.0</td>
      <td>34.0</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>CFO</th>
      <td>1496.0</td>
      <td>20.348930</td>
      <td>3.679270</td>
      <td>2.0</td>
      <td>19.00</td>
      <td>21.0</td>
      <td>23.0</td>
      <td>24.0</td>
      <td>1496.0</td>
      <td>19.959893</td>
      <td>17.268867</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>15.5</td>
      <td>29.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>CTO</th>
      <td>1488.0</td>
      <td>20.129704</td>
      <td>3.695357</td>
      <td>5.0</td>
      <td>18.00</td>
      <td>21.0</td>
      <td>23.0</td>
      <td>24.0</td>
      <td>1488.0</td>
      <td>20.081989</td>
      <td>17.737153</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>29.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>JUNIOR</th>
      <td>20.0</td>
      <td>22.750000</td>
      <td>1.743409</td>
      <td>18.0</td>
      <td>22.00</td>
      <td>23.5</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>20.0</td>
      <td>8.950000</td>
      <td>9.949742</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.5</td>
      <td>13.5</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>MANAGER</th>
      <td>217.0</td>
      <td>21.364055</td>
      <td>2.575022</td>
      <td>14.0</td>
      <td>20.00</td>
      <td>22.0</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>217.0</td>
      <td>14.000000</td>
      <td>11.736300</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>51.0</td>
    </tr>
    <tr>
      <th>SENIOR</th>
      <td>66.0</td>
      <td>22.257576</td>
      <td>1.791524</td>
      <td>16.0</td>
      <td>21.25</td>
      <td>23.0</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>66.0</td>
      <td>10.651515</td>
      <td>10.544401</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>7.5</td>
      <td>15.0</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>VICE_PRESIDENT</th>
      <td>603.0</td>
      <td>20.842454</td>
      <td>3.064650</td>
      <td>9.0</td>
      <td>19.00</td>
      <td>22.0</td>
      <td>23.0</td>
      <td>24.0</td>
      <td>603.0</td>
      <td>16.925373</td>
      <td>14.145638</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>14.0</td>
      <td>24.5</td>
      <td>93.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
high_salary_outliers.describe()
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
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7117.000000</td>
      <td>7117.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>20.004496</td>
      <td>20.815653</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.867752</td>
      <td>18.110057</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>21.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>23.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>24.000000</td>
      <td>98.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
high_salary_outliers.groupby('jobType').mean()
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
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
    </tr>
    <tr>
      <th>jobType</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CEO</th>
      <td>19.475984</td>
      <td>23.017354</td>
    </tr>
    <tr>
      <th>CFO</th>
      <td>20.348930</td>
      <td>19.959893</td>
    </tr>
    <tr>
      <th>CTO</th>
      <td>20.129704</td>
      <td>20.081989</td>
    </tr>
    <tr>
      <th>JUNIOR</th>
      <td>22.750000</td>
      <td>8.950000</td>
    </tr>
    <tr>
      <th>MANAGER</th>
      <td>21.364055</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>SENIOR</th>
      <td>22.257576</td>
      <td>10.651515</td>
    </tr>
    <tr>
      <th>VICE_PRESIDENT</th>
      <td>20.842454</td>
      <td>16.925373</td>
    </tr>
  </tbody>
</table>
</div>



- <span Style = "color:Green">Most of The High Salary Employees Belong to High Ranking Job Types</span>
- <span style = "color:green">The Majority of Them Are CEO/CFO/CTO </span>
- <span style = "color:red">__Some High Salary Employees Belong to Junior Position Employees__ ----> Further Inspection Required </span>


```python
high_salary_juniors = high_salary_outliers[high_salary_outliers['jobType'] == 'JUNIOR']
high_salary_juniors
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
      <th>jobId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
    </tr>
    <tr>
      <th>salary</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>248</th>
      <td>JOB1362684507729</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>BUSINESS</td>
      <td>FINANCE</td>
      <td>23</td>
      <td>8</td>
    </tr>
    <tr>
      <th>246</th>
      <td>JOB1362684435397</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>ENGINEERING</td>
      <td>OIL</td>
      <td>24</td>
      <td>3</td>
    </tr>
    <tr>
      <th>236</th>
      <td>JOB1362685151013</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>BUSINESS</td>
      <td>FINANCE</td>
      <td>19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>232</th>
      <td>JOB1362685195361</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>BUSINESS</td>
      <td>FINANCE</td>
      <td>18</td>
      <td>15</td>
    </tr>
    <tr>
      <th>230</th>
      <td>JOB1362685035221</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>ENGINEERING</td>
      <td>OIL</td>
      <td>24</td>
      <td>29</td>
    </tr>
    <tr>
      <th>228</th>
      <td>JOB1362685204643</td>
      <td>JUNIOR</td>
      <td>MASTERS</td>
      <td>BUSINESS</td>
      <td>OIL</td>
      <td>24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>227</th>
      <td>JOB1362684908426</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>ENGINEERING</td>
      <td>OIL</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>226</th>
      <td>JOB1362684711465</td>
      <td>JUNIOR</td>
      <td>MASTERS</td>
      <td>ENGINEERING</td>
      <td>WEB</td>
      <td>24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>226</th>
      <td>JOB1362684756041</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>ENGINEERING</td>
      <td>OIL</td>
      <td>23</td>
      <td>25</td>
    </tr>
    <tr>
      <th>225</th>
      <td>JOB1362684408909</td>
      <td>JUNIOR</td>
      <td>MASTERS</td>
      <td>COMPSCI</td>
      <td>OIL</td>
      <td>24</td>
      <td>5</td>
    </tr>
    <tr>
      <th>225</th>
      <td>JOB1362684439042</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>COMPSCI</td>
      <td>FINANCE</td>
      <td>24</td>
      <td>0</td>
    </tr>
    <tr>
      <th>225</th>
      <td>JOB1362685262906</td>
      <td>JUNIOR</td>
      <td>MASTERS</td>
      <td>ENGINEERING</td>
      <td>OIL</td>
      <td>22</td>
      <td>26</td>
    </tr>
    <tr>
      <th>225</th>
      <td>JOB1362685093462</td>
      <td>JUNIOR</td>
      <td>BACHELORS</td>
      <td>ENGINEERING</td>
      <td>OIL</td>
      <td>24</td>
      <td>13</td>
    </tr>
    <tr>
      <th>225</th>
      <td>JOB1362685053242</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>BUSINESS</td>
      <td>FINANCE</td>
      <td>24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>223</th>
      <td>JOB1362685362055</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>BUSINESS</td>
      <td>OIL</td>
      <td>24</td>
      <td>26</td>
    </tr>
    <tr>
      <th>223</th>
      <td>JOB1362684568020</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>BUSINESS</td>
      <td>FINANCE</td>
      <td>22</td>
      <td>3</td>
    </tr>
    <tr>
      <th>222</th>
      <td>JOB1362684622293</td>
      <td>JUNIOR</td>
      <td>MASTERS</td>
      <td>BUSINESS</td>
      <td>FINANCE</td>
      <td>22</td>
      <td>4</td>
    </tr>
    <tr>
      <th>222</th>
      <td>JOB1362685064259</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>BUSINESS</td>
      <td>OIL</td>
      <td>22</td>
      <td>3</td>
    </tr>
    <tr>
      <th>221</th>
      <td>JOB1362684597269</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>BUSINESS</td>
      <td>OIL</td>
      <td>24</td>
      <td>11</td>
    </tr>
    <tr>
      <th>221</th>
      <td>JOB1362684835280</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>ENGINEERING</td>
      <td>FINANCE</td>
      <td>23</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



- <span Style = "color:Green">Most of The High Salary Juniors Have Masters and Doctoral Degrees</span>
- <span Style = "color:Green">The Only Employee With Bachelors Degree Employee Has Over 20 Years of Experience</span>
- __Theres No Reason to Suspect that The Data Is Corrupted__

- <span style = "color:red">~~__Some High Salary Employees Belong to Junior Position Employees__ ----> Further Inspection Required ~~</span>

- Checking for Any Logical Errors About Low Payed Employees(salary Below 8.5)


```python
low_salary_outliers = train[train['salary'] < 8.5]
low_salary_outliers
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
      <th>jobId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



- <span style = 'color : green' > There Are No Employees with Salaries Lower than 8.5 </span>

- <span style = 'color:red' >~~Considering the Overall Range It Means that __There Are a Few Extreme Outliers__ ----> Further Inspection Required~~</span>

<a id = featuresDistribution></a>
## EDA 2 - Feature Distribution 


```python
cat_cols = train.drop('jobId',axis=1).select_dtypes(['object']).columns.values
cat_cols
```




    array(['jobType', 'degree', 'major', 'industry'], dtype=object)




```python
plt.figure(figsize=(13, 10))
i = 1
for col in cat_cols:
    plt.subplot(3,3,i)
    
    sns.countplot(train[col])
    plt.xticks(rotation = 45)
    plt.tick_params(labelbottom = True)
    
    i += 1
plt.tight_layout()
```


![png](output_62_0.png)


- <span Style='color:Green' > Jobtype, Degree and Industry Features Are Uniformly Distributed</span>
- <span Style='color:red' > Major Is __Not__ Uniformly Distributed</span>

<a id = FeaturesAnalysis></a>
## EDA 3 - Feature Analysis - Finding Correlation for Each Feature with The Target (Salary)

- <ins> __jobType__ </ins>:


```python
job_salary_df = train[['jobType','salary']]
sorted_train = job_salary_df.sort_values(ascending = False, by = 'salary')
chart = sns.boxplot(x = 'jobType', y = 'salary', data=sorted_train)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
```




    [Text(0, 0, 'CTO'),
     Text(0, 0, 'CFO'),
     Text(0, 0, 'CEO'),
     Text(0, 0, 'VICE_PRESIDENT'),
     Text(0, 0, 'MANAGER'),
     Text(0, 0, 'SENIOR'),
     Text(0, 0, 'JUNIOR'),
     Text(0, 0, 'JANITOR')]




![png](output_66_1.png)



```python
train.groupby('jobType')['salary'].mean().sort_values().plot()
plt.xticks(rotation = 45)
```




    (array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.]),
     <a list of 10 Text xticklabel objects>)




![png](output_67_1.png)


- <span style = "color:green"> Positive Correlation Between Job Type to Salary </span>

- <ins> __Degree__ </ins>:


```python
degree_salary_df = train[['degree','salary']]
sorted_train = degree_salary_df.sort_values(ascending = False, by = 'salary')
chart = sns.boxplot(x = 'degree', y = 'salary', data=sorted_train)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
```




    [Text(0, 0, 'MASTERS'),
     Text(0, 0, 'DOCTORAL'),
     Text(0, 0, 'BACHELORS'),
     Text(0, 0, 'HIGH_SCHOOL'),
     Text(0, 0, 'NONE')]




![png](output_70_1.png)



```python
train.groupby('degree')['salary'].mean().sort_values().plot()
plt.xticks(rotation = 45)
```




    (array([-0.5,  0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5]),
     <a list of 11 Text xticklabel objects>)




![png](output_71_1.png)


- <span style = "color:green"> Positive Correlation Between Degree to Salary </span>

- <ins> __Major__ </ins> :


```python
major_salary_df = train[['major','salary']]
sorted_train = major_salary_df.sort_values(ascending = False, by = 'salary')
chart = sns.boxplot(x = 'major', y = 'salary', data=sorted_train)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
```




    [Text(0, 0, 'ENGINEERING'),
     Text(0, 0, 'BUSINESS'),
     Text(0, 0, 'PHYSICS'),
     Text(0, 0, 'COMPSCI'),
     Text(0, 0, 'BIOLOGY'),
     Text(0, 0, 'CHEMISTRY'),
     Text(0, 0, 'MATH'),
     Text(0, 0, 'LITERATURE'),
     Text(0, 0, 'NONE')]




![png](output_74_1.png)



```python
train.groupby('major')['salary'].mean().sort_values().plot(kind = 'bar')
plt.xticks(rotation = 45)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8]), <a list of 9 Text xticklabel objects>)




![png](output_75_1.png)


- <span style = "color:green"> The Highest Paying Majors Are : Engineering and Business </span>

- <ins> __Industry__ </ins> :


```python
industry_salary_df = train[['industry','salary']]
sorted_train = industry_salary_df.sort_values(ascending = False, by = 'salary')
chart = sns.boxplot(x = 'industry', y = 'salary', data=sorted_train)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
```




    [Text(0, 0, 'OIL'),
     Text(0, 0, 'FINANCE'),
     Text(0, 0, 'WEB'),
     Text(0, 0, 'HEALTH'),
     Text(0, 0, 'AUTO'),
     Text(0, 0, 'SERVICE'),
     Text(0, 0, 'EDUCATION')]




![png](output_78_1.png)



```python
train.groupby('industry')['salary'].mean().sort_values().plot(kind = 'bar')
plt.xticks(rotation = 45)
```




    (array([0, 1, 2, 3, 4, 5, 6]), <a list of 7 Text xticklabel objects>)




![png](output_79_1.png)


- <span style = "color:green"> The Highest Paying Industries Are : Oil, Finance and Web </span>

- <ins>__Experience__</ins>:


```python
year_df = train.groupby('yearsExperience')['salary'].mean()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Experience')
ax1.set_ylabel('Salary')
ax1.set_title('Experience vs Salary')
year_df.plot(kind = 'line')
plt.show()
```


![png](output_82_0.png)


- <span style = "color:green"> Positive Correlation Between Experience to Salary </span>

- <ins>__milesFromMetropolis__<ins>:


```python
distance_df = train.groupby('milesFromMetropolis')['salary'].mean()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('distance')
ax1.set_ylabel('Salary')
ax1.set_title('distance vs Salary')
distance_df.plot(kind = 'line')
plt.show()
```


![png](output_85_0.png)


- <span style = "color:green"> Negetive Correlation Between Miles from Metropolis to Salary </span>

<a id = CorrelationAnalysis></a>
## EDA 4 - Correlation Analysis

<ins> __Giving Categories a Score Value__ 
- Score = Salary Mean   


```python
def encode_label(df, col):
    '''encode the categories using average salary for each category to replace label'''
    cat_dict ={}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = train_avg_salary[train_avg_salary[col] == cat]['salary'].mean()   
    df[col] = df[col].map(cat_dict)
train_avg_salary = train.copy()
```


```python
for col in train_avg_salary.columns:
    if train_avg_salary[col].dtype.name == "object" and col != 'jobId':
        train_avg_salary[col] = train_avg_salary[col].astype('category')
        encode_label(train_avg_salary, col)
        train_avg_salary[col] = pd.to_numeric(train_avg_salary[col])
```


```python
# Correlations between selected features and response
# jobId is discarded because it is unique for individual
fig = plt.figure(figsize=(12, 12))
features = [
 'jobType',
 'degree',
 'major',
 'industry',
 'yearsExperience',
 'milesFromMetropolis']
 
sns.heatmap(train_avg_salary[features + ['salary']].corr(),xticklabels=1 ,cmap='Blues', annot=True)
plt.xticks(rotation=45)
plt.show()
```


![png](output_91_0.png)


<ins> __Positive Correlation with Salary:__
- Jobtype
- Degree
- Major and Experience
- Industry

<ins> __Negetive Correlation with Salary:__
- Miles from Metropolis

<ins> __Other Notes:__
- There's a Positive Correlation Square for The Features : Job Type, Degree and Major
- <span style = 'color:red'> There's a __Very High Correlation__ Between Degree and Major, Such Correlation Will Negatively Affect Some Models</span>

<a id = BaselineModel></a>
## EDA 5 - Establishing a Baseline Model
- The Metric Chosen to Measure Performance Is : MSE

- Let's Consider a Model that Predicts the Salary Based Solely On The Job Type
For Instance, if Mean Salary of Manager Is 100k, Then All the Jobs with Job Type 'manager' Will Have 100k as Their Predicted Salary


```python
train[['jobType','major','degree','industry']] = train[['jobType','major','degree','industry']].astype('category')
```


```python
def baseline(df,col):
    
    jobs_dict = {}
    pred_salary = []
    jobs = train[col].cat.categories.tolist()
    
    for job in jobs:
        jobs_dict[job] = train[train[col] == job]['salary'].mean()
    
    pred_salary = df[col].map(jobs_dict)
    
    return pred_salary
```


```python
baseline(train,'jobType')
```




    0         135.458547
    1         145.311425
    2         125.368630
    3         115.368518
    4         125.368630
                 ...    
    999995    125.368630
    999996    135.481067
    999997     95.333087
    999998    135.458547
    999999     95.333087
    Name: jobType, Length: 999995, dtype: category
    Categories (8, float64): [145.311425, 135.458547, 135.481067, 70.813045, 95.333087, 115.368518, 105.487775, 125.368630]




```python
baseline_score = mse(train['salary'], baseline(train, 'jobType'))
print('baseline_model has an MSE of: ',baseline_score )
```

    baseline_model has an MSE of:  963.9252996562975
    

- Our Aim Is to Achieve a Better Score than __963.9 Mean Squared Error__

# <ins> Pre-processing 

<a id = CategoricalData></a>
## Turning categorical data into numeric values


```python
categorical_vars = ['jobType', 'degree', 'industry','major']
numeric_vars = ['yearsExperience', 'milesFromMetropolis']
target_var = 'salary'
```

<ins> __Turning Ordinal Data(Job Type, Degree) Into Numerical__


```python
cat_df = pd.get_dummies(train[categorical_vars])
num_df = train[numeric_vars]
```

<a id = TrainAndTarget></a>
## Creating Train and Target Data Sets


```python
clean_train_df = pd.concat([cat_df,num_df],axis=1)
target_df = train[target_var]
```


```python
clean_train_df.shape
```




    (999995, 31)




```python
clean_train_df.columns
```




    Index(['jobType_CEO', 'jobType_CFO', 'jobType_CTO', 'jobType_JANITOR',
           'jobType_JUNIOR', 'jobType_MANAGER', 'jobType_SENIOR',
           'jobType_VICE_PRESIDENT', 'degree_BACHELORS', 'degree_DOCTORAL',
           'degree_HIGH_SCHOOL', 'degree_MASTERS', 'degree_NONE', 'industry_AUTO',
           'industry_EDUCATION', 'industry_FINANCE', 'industry_HEALTH',
           'industry_OIL', 'industry_SERVICE', 'industry_WEB', 'major_BIOLOGY',
           'major_BUSINESS', 'major_CHEMISTRY', 'major_COMPSCI',
           'major_ENGINEERING', 'major_LITERATURE', 'major_MATH', 'major_NONE',
           'major_PHYSICS', 'yearsExperience', 'milesFromMetropolis'],
          dtype='object')



# <INS> Develop

<a id = CreatingModels></a>
##  Creating Models

1) <ins> __Linear Regression__<ins>


```python
X = np.array(clean_train_df)
y = np.array(target_df)
X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
```


```python
clf = LinearRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print('Linear Regression:')
accuracy0 = clf.score(X_test,y_test)
print('Regression Accuracy : ' + str(accuracy0))
```

    Linear Regression:
    Regression Accuracy : 0.7438185950862517
    

2) <ins> __RandomForest__<ins>


```python
X = np.array(clean_train_df)
y = np.array(target_df)
X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
```


```python
rf = RandomForestRegressor(n_estimators = 100, n_jobs=-1, max_depth = 25, min_samples_split=60 )
rf.fit(X_train,y_train)
print('Random Forest:')
accuracy0 = rf.score(X_test,y_test)
print('Random Forest Accuracy : ' + str(accuracy0))
```

    Random Forest:
    Random Forest Accuracy : 0.7566824932891807
    

3) <ins> __GradientBoosting__<ins>


```python
X = np.array(clean_train_df)
y = np.array(target_df)
X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
```


```python
gbm = GradientBoostingRegressor(n_estimators=100, max_depth=6, loss='ls',verbose=0)
gbm.fit(X_train,y_train)
print('Gradient Boosting:')
accuracy0 = gbm.score(X_test,y_test)
print('Gradient Boosting Accuracy : ' + str(accuracy0))
```

    Gradient Boosting:
    Gradient Boosting Accuracy : 0.7614124611269681
    


```python
mean_mse = {}
cv_std = {}


neg_mse = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=2, scoring='neg_mean_squared_error')
mean_mse[clf] = -1.0*np.mean(neg_mse)
cv_std[clf] = np.std(neg_mse)

print("Beginning cross validation")
print('LinearRegression() ')
print('Average MSE:\n', mean_mse[clf])
print('Standard deviation during CV:\n', cv_std[clf])
```

    Beginning cross validation
    LinearRegression() 
    Average MSE:
     384.4522213044704
    Standard deviation during CV:
     1.5161618654514382
    


```python
mean_mse = {}
cv_std = {}


neg_mse = cross_val_score(rf, X_train, y_train, cv=5, n_jobs=2, scoring='neg_mean_squared_error')
mean_mse[rf] = -1.0*np.mean(neg_mse)
cv_std[rf] = np.std(neg_mse)

print("Beginning cross validation")
print('RandomForest ')
print('Average MSE:\n', mean_mse[rf])
print('Standard deviation during CV:\n', cv_std[rf])
```

    Beginning cross validation
    RandomForest 
    Average MSE:
     367.0372453106321
    Standard deviation during CV:
     1.1793697694990812
    


```python
mean_mse = {}
cv_std = {}


neg_mse = cross_val_score(gbm, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mean_mse[gbm] = -1.0*np.mean(neg_mse)
cv_std[gbm] = np.std(neg_mse)

print("Beginning cross validation")
print('Gradient Boosting ')
print('Average MSE:\n', mean_mse[gbm])
print('Standard deviation during CV:\n', cv_std[gbm])
```

    Beginning cross validation
    Gradient Boosting 
    Average MSE:
     358.0718742654236
    Standard deviation during CV:
     1.4564274169623386
    

<a id = ModelSelection></a>
## Selecting the best model

- The Model with The Lowest Mean Square Error(MSE) Will Be the Best Model.


```python
model = min(mean_mse, key=mean_mse.get)
print('\nBest model is: \n')
print(model)
```

    
    Best model is: 
    
    GradientBoostingRegressor(max_depth=6)
    

- We Need to Fit the __Gradient Boosting Regressor__ Model on All the Training Data to Make Future Salary Predictions from The Test Dataset


```python
model.fit(X,y)
```




    GradientBoostingRegressor(max_depth=6)



# <ins> Deploy

- The Test Data Should Be Cleaned as The Train Data Was Cleaned and It Should Undergo the Same Encoding and Manipulation to Match the Shape of The Training Dataset

<a id = TestData ></a>
<ins> __The Changes I Have Made to The  Training Dataset Are:__
- Dropping Company Id Column
- Dropping job Id Column
- Turning Categorical Features Into Numeric via Dummy Variables


```python
test_feat = test_features.copy()
```


```python
test_feat = test_feat.drop('companyId',axis=1)
test_feat = test_feat.drop('jobId',axis=1)
```


```python
categorical_vars = ['jobType', 'degree', 'industry','major']
numeric_vars = ['yearsExperience', 'milesFromMetropolis']
```


```python
cat_df = pd.get_dummies(test_feat[categorical_vars])
num_df = test_feat[numeric_vars]

clean_test_df = pd.concat([cat_df,num_df],axis=1)
```


```python
clean_test_df.head()
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
      <th>jobType_CEO</th>
      <th>jobType_CFO</th>
      <th>jobType_CTO</th>
      <th>jobType_JANITOR</th>
      <th>jobType_JUNIOR</th>
      <th>jobType_MANAGER</th>
      <th>jobType_SENIOR</th>
      <th>jobType_VICE_PRESIDENT</th>
      <th>degree_BACHELORS</th>
      <th>degree_DOCTORAL</th>
      <th>...</th>
      <th>major_BUSINESS</th>
      <th>major_CHEMISTRY</th>
      <th>major_COMPSCI</th>
      <th>major_ENGINEERING</th>
      <th>major_LITERATURE</th>
      <th>major_MATH</th>
      <th>major_NONE</th>
      <th>major_PHYSICS</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>22</td>
      <td>73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>14</td>
      <td>96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
#predicting the salaries from test dataset
predictions = pd.DataFrame(model.predict(clean_test_df))
```


```python
#Concatenating the predicted salaries to the original test dataset
prediction_df=pd.concat([test_features,predictions],axis=1)
```

## Exporting The Predictions to A Csv File


```python
prediction_df.to_csv('salaries_predicted.csv')
```

<a id = FeatureImportance></a>
## Feature Importances


```python
importance = model.feature_importances_
feature_importances = pd.DataFrame({'feature':clean_test_df.columns, 'importance':importance})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
#set index to 'feature'
feature_importances.set_index('feature', inplace=True, drop=True)
```


```python
fig = feature_importances[0:27].plot.bar(figsize=(16,10))
fig.set_title('FEATURE IMPORTANCE PLOT')
```




    Text(0.5, 1.0, 'FEATURE IMPORTANCE PLOT')




![png](output_142_1.png)

