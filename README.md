# wbgt-flaskapi

The wet-bulb globe temperature (WBGT) is a type of apparent temperature used to estimate the effect of temperature, humidity, wind speed (wind chill), and visible and infrared radiation (usually sunlight) on humans. It is used by industrial hygienists, athletes, sporting events and the military to determine appropriate exposure levels to high temperatures. It is derived from the following formula: WBGT=0.7Tw+0.2Tg+0.1T

True WBGT is hard to calculate especially without specific variables like Globe thermometer temperature. As such instead we would be looking at deriving WBGT from just 2 easy obtainable variables such as Temperature and Humidity.
In this example we will be making use of NEA's "Realtime Weather Readings across Singapore" API to obtain these variables. 
We can then make use of a WBGT table below to find the associated WBGT value using extrapolation which is our main aim of this program.

![WBGT Chart](https://github.com/coldoasis/wbgt-flaskapi/assets/124854971/4a03cf4f-a253-40ef-9bd4-b195b0089b77)

If you were to try attempt creating a extrapolation method to obtain WBGT for a given temperature and humidity, the return would often be sluggish and takes too long.
Given that we are aware WBGT follows a linear like relationship with its variables, we attempted to make use of a Linear Regression Model to create a relatively accurate linear relationship.
We managed to obtain the following with the linear regression model strategy:

1. Mean Squared Error: 0.36328239934303946
2. R-squared: 0.9856824726635929
3. Linear formula: WBGT = 1.29 * temperature + 0.18 * humidity + -18.53

If you are interested here is the 3D plot of the model:

![WBGT linear model](https://github.com/coldoasis/wbgt-flaskapi/assets/124854971/9afb16d3-ef67-41c0-9616-91c246941c0b)

## Script.py

This code assumes that you will be obtaining data from a DB (MySQL or MsSQL). Do appropriately switched out variables in the configuration.ini.
The code mainly does the following:
1. Obtain required dataset using SQL selection and transforming it into Pandas DataFrame. 
2. Do appropriate preprocessing to remove erroneous datasets: duplication, missing values.
3. Incorporation of WBGT formula into new DF column.
4. Other respective specific filtering of SQL dataset.

## Model.py

### Model creation

Given the temporal nature of WBGT, Long Short-Term Memory (LSTM) models are good fit to our requirements.
This mainly focuses on creation of the model and the saving of it as a .h5 format file for future retraining or loading. h5 format is more compact, stores model architecture and weights alone with related information such as training history and configuration.
Given that there are many weather stations that the provided NEA API consists of, we will have one LSTM model for each of the stations. 

### Model prediction

Prediction for LSTM model is alittle different. From our understanding to forecast/predict WBGT for timeframe 2hrs ahead of current time involves providing dataset obtain from 2hrs prior of current time.
As such code consist of manipulating of dataset such that we can achieve this. User can input X hours, and the program would appropriately filter the DF to obtain the necessary datasets for input of prediction.

## App.py

We will be using Flask to deploy our test environment. Mapping the various endpoints which take in the necessary arguments.

In order to deploy it onto Cloud Services such as Azure in this demo, we should containerise our code using Docker. 
We can do so by creating a Dockerfile (.dockerignore should you wish to exclude some files).



