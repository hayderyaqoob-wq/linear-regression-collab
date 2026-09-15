# src/main.py
import argparse
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Train Linear Regression and predict Y for a given X.")
    parser.add_argument("--x", type=float, help="The input value X to predict Y")
    args = parser.parse_args()

    print("🚀 Starting Linear Regression Training...")
    
    # 1. Generate synthetic dataset (y = ~3.5*X + ~2.0 + noise)
    X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
    
    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Evaluate the model
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    print("\n--- Model Results ---")
    print(f"Coefficient (Weight): {model.coef_[0]:.4f}")
    print(f"Intercept (Bias):     {model.intercept_:.4f}")
    print(f"Mean Squared Error:   {mse:.4f}")
    print(f"R-squared Score:      {r2:.4f}")
    print("---------------------\n")
    
    # 5. Predict for a given X if provided
    if args.x is not None:
        # Reshape to 2D array as expected by scikit-learn
        x_input = np.array([[args.x]])
        y_predicted = model.predict(x_input)[0]
        print(f"🎯 PREDICTION: For x = {args.x}, the predicted y = {y_predicted:.4f}")
    else:
        print("💡 Tip: Run with '--x <value>' to predict a specific Y value (e.g., python src/main.py --x 5.0)")

if __name__ == "__main__":
    main()