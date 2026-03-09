import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# Load data function
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Train model function
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"Mean Absolute Error from Cross-Validation: {-cv_scores.mean()}")
    model.fit(X, y)
    return model

# Main function
if __name__ == "__main__":
    # Load data
    data = load_data('mobile_data.csv')

    # Prepare features and target
    X = data.drop('price', axis=1)  # Assuming 'price' is your target column
    y = data['price']

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Save the model
    with open('mobile_price_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    print("Model saved as 'mobile_price_model.pkl'")
