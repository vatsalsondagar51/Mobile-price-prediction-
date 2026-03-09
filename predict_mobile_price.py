import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load the dataset
data = pd.read_csv('mobile_data.csv')

# Features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Define categorical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline that first preprocesses the data then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)



# Save the model
joblib.dump(pipeline, 'mobile_price_model.pkl')

# Function to predict price based on user input
def predict_price(user_input, model):
    # Create a DataFrame from user input
    input_df = pd.DataFrame(user_input, index=[0])
    
    # Make prediction
    predicted_price = model.predict(input_df)
    
    return predicted_price[0]  # Return the predicted price

# Load the model
loaded_model = joblib.load('mobile_price_model.pkl')

# Collect specifications from the user
print("Enter the specifications for the mobile phone:")
user_input = {}

for feature in X.columns:
    user_input[feature] = input(f"{feature}: ")

# Make prediction based on user input
predicted_price = predict_price(user_input, loaded_model)
print(f"Predicted Price: {predicted_price}")
