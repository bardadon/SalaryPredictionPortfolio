# Salary Prediction Project

## THE PROBLEM

For this Salary prediction portfolio I examined 1 million job records along with their features such as **Job Type, Major, Degree, Industry, Years of Experience, Distance from the metropolis area** and the given **SALARY**. Based on this data, I'm going to build some predictive models and choose the best model (having lowest MSE) and use it to predict salaries of another 1 million jobs based on their features.

### PROJECT OUTLINE:

- Load the data, Understand the data, Find relationships between attributes

- Establish a baseline model, Engineer features, Optimize the data features

- Select the best model with the lowest evaluation metric and

- **Predict salaries of another 1 million job features using the selected model**

## DISCOVER DATA

### Sample Of The Data

|     jobId      | companyId |    jobType   |  degree  |  major  | industry| yearsExperience | milesFromMetropolis| salary
|----------------|------------|--------------|-----------|---------|---------|------------------|--------------------|--------|
|JOB1362684407687|COMP37      |CFO           |MASTERS    |MATH     |HEALTH   |10                |83                  |130     |
|JOB1362684407688|COMP19      |CEO           |HIGH_SCHOOL|NONE     |WEB      |3                 |73                  |101     |
|JOB1362684407689|COMP52      |VICE_PRESIDENT|DOCTORAL   |PHYSICS  |HEALTH   |10                |38                  |137     |
|JOB1362684407690|COMP38      |MANAGER       |DOCTORAL   |CHEMISTRY|AUTO     |8                 |17                  |142     |
|JOB1362684407691|COMP7       |VICE_PRESIDENT|DOCTORAL   |PHYSICS  |FINANCE  |8                 |16                  |163     |

Target Variable (Salary) Distribution

<img src="Plots/Salary Distribution.png"/>

### FEATURES COUNT

<img src = "Plots/Feature Distribution.png"/>

### CATEGORICAL VARIABLES VS AVERAGE SALARY

<img src="Plots/Degree Distribution.png" width="250"/> &ensp;&ensp;&ensp; <img src="Plots/Industry Distribution.png" width="250"/> 

<img src="Plots/Job Type Distribution.png" width="250"/> &ensp;&ensp;&ensp; <img src="Plots/Major Distribution.png" width="250"/>

- The job_type CTO has the highest salary distribution, and the Janitor has the lowest.

- The degrees 'Doctoral' and 'Masters' have a higher salary distribution and people with no major (i.e. "NONE") have the lowest

- All the majors have more or less the same salary distribution. Although Engineering major seems to have a little edge over Business major. None major has the lowest distribution.

- Industries 'Oil' and 'Finance' seems to have the highest salary distribution than others.

### CORRELATION: NUMERICAL VARIABLES VS SALARY

<img src="Plots/Experience And Salary Correlation.png" width="450"/>  <img src="Plots/Miles from Metropolis And Salary Correlation.png" width="450"/>

- The yearsExperience variable has a positive and steady relationship with salary which means, as the years of experince increase the salary will also increase.

- The milesFromMetropolis variable has a negative and steady relationship with salary which means, as the job location is further away from the metropolis the salary will go on decreasing.

### CORRELATION MATRIX

<img src="Plots/Correlation Matrix.png" />

- Positive correlation between variables **job_Type, degree, major, industry, yearsExperience** and **Salary**

- Negative correlation between variable **milesFromMetropolis** and **Salary**

- Strong positive correlation between **degree** and **major** 

- Weak positive correlation between **job_Type** and **degree, major** 

## BASELINE MODEL

I created a baseline model which predicted salaries for every record based on the **mean** of that specific **job_type**.

After prediction, I used Mean_squared_error as an evaluation metric and got the following result :

|Model|MSE_Score|

|-----|-----|
|Baseline_model|963.92|

## DEVELOP MODEL

- Categorical variables: **job_Type, degree, major, industry** - one-hot-encoding them via dummy variables.

- Numerical variables: **yearExperience , milesFromMetropolis**.

- Target variable: **salary**.

### Based on the above EDA, I chose the following predictors for modeling

| VARIABLE      | TYPE          |Columns|
| ------------- | ------------- |-------|
| yearsExperience  | INT  |1|
| milesFromMetropolis  | INT  |1|
|job_Type - CFO, CTO, CEO, Janitor, Junior, Manager, Senior, Vice-president |CAT - INT| 8|
|degree - None, High_school, Bachelors, Masters, Doctral |CAT - INT|5|
|major - None, Biology, Business, Chemistry, CompSci, Literature, Math, Physics, Engineering |CAT - INT|9|
|industry - Education, Finance, Health, Oil, Service, Web, Auto |CAT - INT|7|
|**Total number of predictors**||**31**|

Final dataset comprises 9999995 rows and 31 columns.

I did a 80% training and 20% testing split on this dataset. I applied the Following models after feature engineering and collected their Mean_squared_error:

|No.|Model|MSE_Score|Standard-deviation|
|---|-----|---------|------------------|
|1. |Linear Regression|384.45|1.51|
|2. |Random Forest Regressor|367.03|1.17|
|3. |Gradient Boosting Regressor|358.07|1.45|

#### Gradient Boosting Regressor provides the best results, so I used this model to predict the salaries of the test data and saved it in a csv file.

The Plot below shows the feature importances of the features used in the model

<img src="Plots/Features Importance.png"/> 

# THANK YOU!
