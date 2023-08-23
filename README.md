# wbgt-flaskapi

The wet-bulb globe temperature (WBGT) is a type of apparent temperature used to estimate the effect of temperature, humidity, wind speed (wind chill), and visible and infrared radiation (usually sunlight) on humans. It is used by industrial hygienists, athletes, sporting events and the military to determine appropriate exposure levels to high temperatures. It is derived from the following formula: WBGT=0.7Tw+0.2Tg+0.1T

True WBGT is hard to calculate especially without specific variables like Globe thermometer temperature. As such instead this demo aims to look at deriving WBGT from just 2 easy obtainable variables such as Temperature and Humidity.
In this example we will be making use of NEA's "Realtime Weather Readings across Singapore" API to obtain these variables. https://beta.data.gov.sg/datasets/1459/view

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

```python
def trainModel(station_id):
    # 1 Model for each station
    df = getDataFrame(station_id)

    # Split the data into input features (X) and target variable (y)
    X = df[['temperature', 'humidity']].values
    y = df['WBGT'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the input features for LSTM (assuming you have a time step of 1)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(25, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    model_filename = f'model_{station_id}.h5'

    model.save(model_filename)

```

Given that there are many weather stations that the provided NEA API consists of, we will have one LSTM model for each of the stations. 

### Model prediction

Prediction for LSTM model is alittle different. From our understanding to forecast/predict WBGT for timeframe 2hrs ahead of current time involves providing dataset obtain from 2hrs prior of current time.

As such code consist of manipulating of dataset such that we can achieve this. User can input X hours, and the program would appropriately filter the DF to obtain the necessary datasets for input of prediction.

```python

def predictModel(model,hour,df):
    current_time = datetime.now()

    print(current_time)

    # Calculate the end point of the 5-hour interval from the current time
    end_time = current_time - timedelta(hours=hour)

    print(end_time)

    print(df.tail())

    # Convert timestamp column to datetime if needed
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter the DataFrame to get rows within the 5-hour interval (Include the current time as well)
    new_df = df[df['timestamp'] > end_time]

    print(new_df)

    # Calculate the number of rows to select (approximately 4 per hour)
    num_rows_to_select = 4 * hour
    num_rows = len(new_df)

    # Select rows at regular intervals
    interval = max(1, num_rows // num_rows_to_select)  # Ensure at least 1 row selected
    selected_indices = range(0, num_rows, interval)
    selected_rows = new_df.iloc[selected_indices]

    X = selected_rows[['temperature', 'humidity']].values

    print(X.shape)

    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    # Predict using the model
    predictions = model.predict(X)

    # Create a new DataFrame to store the timestamp and predictions
    timestamp_values = selected_rows['timestamp'].values
    predicted_values = predictions.reshape(-1)
    
    # Create a list to store the timestamp and predictions as dictionaries
    result_list = []
    for timestamp, prediction in zip(selected_rows['timestamp'], predictions):
        # Convert timestamp to string in the desired format (adjust format as needed)

        future_timestamp = timestamp + timedelta(hours=hour)

        timestamp_str = future_timestamp.strftime('%Y-%m-%d %H:%M:%S')

        # Append the timestamp and prediction as dictionaries to the result_list
        result_list.append({'timestamp': timestamp_str, 'predicted_value': str(prediction[0])})

    # Convert the result_list to JSON format
    result_json = json.dumps(result_list)

    return result_json

```

## App.py

We will be using Flask to deploy our test environment. Mapping the various endpoints which take in the necessary arguments.

In order to deploy it onto Cloud Services such as Azure in this demo, we should containerise our code using Docker. 
We can do so by creating a Dockerfile (.dockerignore should you wish to exclude some files).

```python
COPY requirements.txt .

RUN pip3 install -r requirements.txt
```
Here we specify the program to pip install all the required dependencies listed in requirements.txt

Finally, we will be using Gunicorn as a means to deploy our overall Flask Web Server.

```python
ENTRYPOINT ["gunicorn", "app:app"]
```
We set the appropriate gunicorn configurations in gunicorn.conf.py.

## Azure Deployment

1. Create a web app service on Azure.
2. For deployment select either DockerHub or Azure Container Registry which contains your docker image.
3. Allow the web app to cold boot.
4. Test API endpoints by using provided url generated by Azure.
