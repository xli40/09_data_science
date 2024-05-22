## setup ##
import pandas as pd
import numpy as np

# 1: Read the file airlines.xlsx in a dataframe df_data.
df_data = pd.read_excel('airlines.xlsx')

# 2: Clean the data in df_data. This means remove all rows that have at least one empty value.
df_data = df_data.dropna()

# 3: Write a line of code to double check you do not have any null values in df_data.
assert df_data.isnull().sum().sum() == 0, "null found"

# 4: Convert the columns that store date and time from string (object) to data and time respectively.
df_data['Date_of_Journey'] = pd.to_datetime(df_data['Date_of_Journey'], format='%d/%m/%Y')
df_data['Dep_Time'] = pd.to_datetime(df_data['Dep_Time']).dt.time
df_data['Arrival_Time'] = pd.to_datetime(df_data['Arrival_Time']).dt.time

# 5: Add the following columns to df_data: 'Journey_day', 'Journey_month', 'Journey_year', 'Dep_Time_hour', 'Dep_Time_minute', 'Arrival_Time_hour', 'Arrival_Time_minute' and fill it with the corresponding data.

# 6: Create a new column 'dep_description' where depature time between:
#    4am and 8am returns "Early Morning"
#    between 8am and 12pm return "Morning"
#    between 12pm and 4pm return "Noon"
#    between 4pm and 8pm return "Evening"
#    between 8pm and 12am return "Nigth"
#    between 12 am and 4am return "Late Night"

# 7 ? Does not exist

# 8: Build a bar chart where the x-axis represents the 'dep_description' and the y-axis the corresponding number of flights.

# 9: Create 3 new columns: 'Duration_hours', 'Duration_mins' and 'Duration_total_mins'.

# 10: Build a scatter plot with x-axis: 'Duration_total_mins' and y-axis 'Price'.

# 11: Let's add a simple regression line on top of the chart you created on number 10. For this regression use: numpy.polyfit with degree 1 Analyze the result (relationship between variables) and write your comments about it.

# 12: Build a scatter plot with x-axis 'Duration_total_mins' and y-axis 'Price', but add different colors for the different number of stops. Analyze the result and write your comments about it.

# 13: Determine which route Jet Airways is the most used (write yor answer on the notebook). Create a bar chart with these most used routes (for Jet Airways) in descending order: x-axis must represent the routes, and y-axis must represent the number of routes.

# 14: Based on the regression lesson https://github.com/novillo-cs/softdev_material/blob/main/projects/09_data_science/regression/scikit.ipynb
#     Replicate the regression you made on number 11. Now, you must use patsy and sklearn to create the model and apply the regression. Find the score (r2) and comment about the result.

# 15: Implement a linear regression using patsy and sklearn for the following model and find the score(r2). Write your comments about this model and the score. Price ~ Airline * Source * Destination * Total_Stops * Dep_Description * Journey_month * Journey_weekday

# 16: Try a different model to see if you have a better r2. Write your comments about the result, and compare this result with the one in the previous exercise.
