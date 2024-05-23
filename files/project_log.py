## setup ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import patsy 
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1: Load the data from the Excel file 'airlines.xlsx' into a DataFrame named df_data.
df_data = pd.read_excel('airlines.xlsx')
display(df_data)

# 2: Remove any rows that contain missing values to ensure a clean dataset.
df_data.dropna(inplace=True)

# 3: Verify that there are no remaining null values in df_data.
assert df_data.isnull().sum().sum() == 0, "Null exists"

# 4: Convert the 'Date_of_Journey', 'Dep_Time', and 'Arrival_Time' columns from strings to datetime objects.
df_data['Date_of_Journey'] = pd.to_datetime(df_data['Date_of_Journey'], format='%d/%m/%Y')
df_data['Dep_Time'] = pd.to_datetime(df_data['Dep_Time'])
df_data['Arrival_Time'] = pd.to_datetime(df_data['Arrival_Time'])
display(df_data)

# 5: Create new columns for the day, month, and year of the journey, as well as the hour and minute of the departure and arrival times.
df_data['Journey_day'] = df_data['Date_of_Journey'].dt.day
df_data['Journey_month'] = df_data['Date_of_Journey'].dt.month
df_data['Journey_year'] = df_data['Date_of_Journey'].dt.year
df_data['Dep_Time_hour'] = df_data['Dep_Time'].dt.hour
df_data['Dep_Time_minute'] = df_data['Dep_Time'].dt.minute
df_data['Arrival_Time_hour'] = df_data['Arrival_Time'].dt.hour
df_data['Arrival_Time_minute'] = df_data['Arrival_Time'].dt.minute
display(df_data)

# 6: Create a new column 'dep_description' that categorizes the departure time into different parts of the day.
def classify_departure(hour):
    if 4 <= hour < 8: return "Early Morning"
    elif 8 <= hour < 12: return "Morning"
    elif 12 <= hour < 16: return "Noon"
    elif 16 <= hour < 20: return "Evening"
    elif 20 <= hour < 24: return "Night"
    else: return "Late Nigth"
df_data['dep_description'] = df_data['Dep_Time_hour'].apply(classify_departure)
display(df_data['dep_description'])

# 7 ? 

# 8: Plot a bar chart showing the number of flights for each departure time category.
departure_counts = df_data['dep_description'].value_counts().reindex(["Early Morning", "Morning", "Noon", "Evening", "Night", "Late Night"], fill_value=0)
plt.figure(figsize=(10,6))
departure_counts.plot(kind='bar')
plt.title('Number of Flights by Departure Time Category')
plt.xlabel('Departure Time Category')
plt.ylabel('Number of Flights')
plt.xticks(rotation=0)
plt.show()

# 9: Create three new columns to represent the duration of the flights in hours, minutes, and total minutes.
def extract_duration(duration):
    hours, minutes = 0, 0
    if 'h' in duration:
        hours = int(duration.split('h')[0])
        duration = duration.split('h')[1]
    if 'm' in duration: minutes = int(duration.split('m')[0])
    return hours, minutes
df_data[['Duration_hours', 'Duration_mins']] = df_data['Duration'].apply(lambda x: pd.Series(extract_duration(x)))
df_data['Duration_total_mins'] = df_data['Duration_hours'] * 60 + df_data['Duration_mins']

# 10: Create a scatter plot of flight duration (in minutes) against flight price.
plt.figure(figsize=(10,6))
plt.scatter(df_data['Duration_total_mins'], df_data['Price'], alpha=0.5, label='Data Points')

# 11: Fit a simple linear regression line to the scatter plot using numpy.polyfit and analyze the results.
plt.figure(figsize=(10,6))
plt.scatter(df_data['Duration_total_mins'], df_data['Price'], alpha=0.5, label='Data Points')
slope, intercept = np.polyfit(df_data['Duration_total_mins'], df_data['Price'], 1)
regression_line = slope * df_data['Duration_total_mins'] + intercept
plt.plot(df_data['Duration_total_mins'], regression_line, color='red', label='Regression Line')
plt.title('Flight Duration vs Price')
plt.xlabel('Total Duration (min)')
plt.ylabel('Price')
plt.legend()
plt.show()
print(f"Regression Line Equation: Price = {slope:.2f} * Duration_total_mins + {intercept:.2f}")

# 12: Create a scatter plot of flight duration against price, colored by the number of stops.
unique_stops = df_data['Total_Stops'].unique()
color_map = plt.cm.viridis(np.linspace(0, 1, len(unique_stops)))
plt.figure(figsize=(10,6))
for stop, color in zip(unique_stops, color_map):
    subset = df_data[df_data['Total_Stops'] == stop]
    plt.scatter(subset['Duration_total_mins'], subset['Price'], label=stop, alpha=0.6, color=color)
plt.title('Flight Duration vs Price by Number of Stops')
plt.xlabel('Duration (minutes)')
plt.ylabel('Price')
plt.legend(title='Total Stops')
plt.show()
print("The distrbution of flight durations and prices for different numbers of stops, but the majority of flights have only one stop.")

# 13: Identify the most frequently used routes by Jet Airways and visualize the top 10 routes in a bar chart.
jet_airways_routes = df_data[df_data['Airline'] == 'Jet Airways']['Route'].value_counts().head(10)
plt.figure(figsize=(12,6))
jet_airways_routes.plot(kind='bar')
plt.title('Top 10 Most Used Routes by Jet Airways')
plt.xlabel('Route')
plt.ylabel('Number of Flights')
plt.xticks(rotation=45)
plt.show()

# 14: Perform linear regression using patsy and sklearn on the formula 'Price ~ Duration_total_mins' and evaluate the model's performance.
y, X = dmatrices('Price ~ Duration_total_mins -1', data=df_data, return_type='dataframe')
simple_model = LinearRegression()
simple_model.fit(X, y)
y_pred = simple_model.predict(X)
simple_r2 = r2_score(y, y_pred)
print(f'R² score for Price ~ Duration_total_mins: {simple_r2:.3f}')
print("The R² score indicates the proportion of variance in the price that can be explained by the duration of the flight. The obtained score is 0.257, which suggests a moderate correlation.")

# 15: Implement a linear regression model using patsy and sklearn with a more complex formula and evaluate its performance.
formula = "Price ~ Airline + Source + Destination + Total_Stops + C(Journey_month) + Journey_day -1"
y, X = dmatrices(formula, data=df_data, return_type='dataframe')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
simplified_model = LinearRegression()
simplified_model.fit(X_train, y_train)
training_score = simplified_model.score(X_train, y_train)
print(f'R² score on training set: {training_score:.3f}')
print('Model coefficients:', simplified_model.coef_[0])
print('Model intercept:', simplified_model.intercept_[0])

# 16: Try a different model to see if you have a better r2. Write your comments about the result, and compare this result with the one in the previous exercise.
alternative_formula = 'Price ~ Duration_total_mins + Airline + Source + Destination + Total_Stops -1'
y, X = dmatrices(alternative_formula, data=df_data, return_type='dataframe')
alternative_model = LinearRegression()
alternative_model.fit(X, y)
y_pred_alternative = alternative_model.predict(X)
alternative_r2 = r2_score(y, y_pred_alternative)
print(f'R² score for alternative model: {alternative_r2:.3f}')
print("The alternative model, which includes more variables, achieves an R² score of 0.601, indicating a significantly better fit compared to the simpler model. This suggests that these additional variables have a substantial impact on the price of flights.")