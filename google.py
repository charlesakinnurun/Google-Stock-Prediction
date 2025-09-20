# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

# %% [markdown]
# Load the dataset

# %%
df = pd.read_csv("google.csv")

# %% [markdown]
# Display first 5 rows and the dataset information

# %%
print(df.head())
print(df.info())

# %% [markdown]
# Rename the columns for clarity and consistency

# %%
df.rename(columns={
    "Date":"date",
    "Open":"open",
    "High":"high",
    "Low":"low",
    "Close":"close",
    "Adj Close":"adj_close",
    "Volume":"volume"
},inplace=True)
print(df.info())

# %% [markdown]
# Convert the date column to datetime objects

# %%
df["date"] = pd.to_datetime(df["date"])

# %% [markdown]
# Create a new feature "day" representing the number of days since the first day in the dataset

# %%
df["day"] = (df["date"] - df["date"].min()).dt.days

# %% [markdown]
# Drop the original date column as we won't need need it anymore

# %%
df = df.drop("date",axis=1)
print(df.head().to_string())

# %% [markdown]
# Define Features (X) and target (y)

# %%
# Our feature (X) is the "day", which will use to predict the "close" price (y)
# We need to reshape X to a 2D array because sckit-learn models expect it that way
X = df[["day"]]
y = df[["close"]]

# %% [markdown]
# Split the data into training and testing sets

# %%
# We use 80% of the data for training the model and 20% for testing its performance
# random_state ensures that the split is the same every time the script is run
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Shape of the training data:",X_train.shape)
print("Shape of testing data:",X_test.shape)

# %% [markdown]
# Choose and train the model

# %%
# We'll use a simple Linear Regression model, which is great for beginners
model = LinearRegression()

# %% [markdown]
# Train the model using the training data

# %%
model.fit(X_train,y_train)

# %% [markdown]
# Make predictions on the test set

# %%
y_pred = model.predict(X_test)

# %% [markdown]
# Evaluate the model's performance

# %%
# We'll use two common metrics: Mean Squared Error and R-squared
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

# Model Evaluation
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")
print(f"Model's slope (coefficent): {model.coef_[0]}")
print(f"Model's y-intercept: {model.intercept_}")

# %% [markdown]
# Use the trained model to make a new prediction

# %%
# Let's predict the closing price for a hypothetical future date, for example from the start
days_in_20_years = 20 * 365  # Use 365 for calendar days, or 252 for trading days
future_date_prediction = model.predict([[days_in_20_years]])
print(f"\nPredicted closing price after {days_in_20_years} days: ${future_date_prediction[0][0]:.2f}")

# %%
# Let's also create a simple visualization to see how well our model fits the data
import matplotlib.pyplot as plt

# Create a scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.title('Actual vs Predicted Close Prices')
plt.show()

# Print some additional insights
print(f"\nModel Performance Summary:")
print(f"- The model explains {r2*100:.1f}% of the variance in stock prices")
print(f"- Average prediction error: ${np.sqrt(mse):.2f}")
print(f"- The stock price increases by approximately ${model.coef_[0][0]:.4f} per day on average")



