# Rental Bike Demand Prediction

For a bike renting system to smoothly function, it is necessary to provide a stable supply of rental bikes at any given point of time according to the demand. This requires having a good prediction of the bike demand at each hour. I am working with a dataset of bike rental counts in the city of Seoul, South Korea which contains historical data on date and weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall).

## Overview
* The distributions of the features as well as their relationship with the rented bike count is explored.
* Linear regression model is trained on the data to make predictions and its performance is evaluated.
* Decision Tree Regression model is trained for getting better predictions and this model's performance is evaluated as well.
* Python libraries such as Matplotlib, Seaborn, Pandas and Scikit-learn are used.

## Objective
The aim is to predict the demand of rental bikes at any given hour using the weather and date information provided in the dataset.

#### Data Source
The dataset was obtained from UCI Machine Learning Repository. [CLICK HERE](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand)

Relevant papers cited by the UCI Machine Learning Repository [[1]](#1) [[2]](#2).

#### Data Preparation
* The dataset column names were simplified.
* Date string values were converted to datetime format in order to retireve month and year information.
* There were no missing values to deal with in the dataset.

## Exploratory Data Analysis
* Distribution of data across categories of various features was studied.
* Distribution of total rented bike counts across categories of features was explored.
* Relationships between the dependent variable and numerical features were visualised.
* Distributions of numerical features were visualised.
* Correlation between dependent and independent variables were calculated.

## Model Fitting
#### Linear Regression Model
* Label encoding and one hot encoding was done on categorical variables.
* Rainfall and snowfall variables were ignored as their relationship with the dependent variable is not linear.
* The dataset was split into training and testing dataset and scaled.
* The model was fit and the target variable predictions were made.
* Model performance was evaluated.

#### Decision Tree Regressor
* label encoding was done as Scikit-learn decision tree regressor does not support categorical variables.
* GridSearchCV was used for hyperparameter tuning and cross validation.
* The model was fit and the target variable predictions were made.
* Model performance was evaluated.
* The best hyperparameter combination (as given by GridSearchCV) was used to train a Decision Tree model in order to visualise the Tree.

## Conclusions
#### Exploratory Data Analysis
* June followed by July and May (summer season) has the most bikes rented. January, February and December (winter) has the least number of bikes rented.
* The peak time of the day rented bike count is around 4-7 pm in the evening. There is a smaller peak in the morning (around 7-9 am). The least activity is during the early morning period.
* Summer is the most active season and winter is the least active one.
* Very low number of bikes were rented during holidays.
* There was a weak linear relationship with some of the variables. However, there was high standard deviation and high heteroskedasticity.
* Rainfall and snowfall variables had a non linear relationship with rented bike counts.
* The distribution of the dependent variable was skewed.
#### Linear Regression
* The model had an R2 score (Coefficient of determination) of 0.51 on testing set and 0.53 on training set i.e the model is able to explain around 51% of the variation in the predicted variable.
* The root mean squared error was found to be 454.37 on testing set and 436.90 on training set.
* The performance of the model on testing and training datset is fairly similar.
#### Decision Tree Regression
* The hyperparameter tuned model gave an R2 score of 0.779 on testing set and 0.832 on training set.
* The root mean squared error was 304.42 on testing set and 261.22 on training set.
* The best parameters were 'max_depth': 7, 'max_leaf_nodes': None, 'min_samples_leaf': 8.

## References
<a id="1">[1]</a> 
Sathishkumar V E, Jangwoo Park, and Yongyun Cho (2020). 
'Using data mining techniques for bike sharing demand prediction in metropolitan city.' Computer Communications, Vol.153, pp.353-366.

<a id="2">[2]</a> 
Sathishkumar V E and Yongyun Cho (2020). 
'A rule-based model for Seoul Bike sharing demand prediction using weather data' European Journal of Remote Sensing, pp. 1-18.
