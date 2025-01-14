import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "employee_id": ["EM_" + str(i) for i in range(101, 201)],
    "experience_years": [
        16.8, 10.7, 14.1, 9.1, 8.9, 7.9, 4.4, 16.2, 2, 0, 3.6, 6.1, 14.7, 6.7, 18.2, 2.8, 15.4, 15.6, 2.4, 6.3,
        11.1, 17.8, 5.3, 8.5, 12.9, 3, 8.1, 19.4, 1.8, 14.5, 2.2, 9.5, 17.2, 5.7, 18.8, 1, 13.3, 13.1, 7.1, 3.2,
        9.9, 6.9, 3.6, 19.2, 5.5, 3.8, 16.4, 5.1, 12.5, 2.6, 4.8, 0.6, 3.4, 7.7, 1.6, 15.8, 1.2, 12.9, 7.3, 18, 11.3,
        20, 10.9, 8.7, 10.1, 13.5, 9.3, 13.7, 12.3, 19.6, 16, 8.3, 11.7, 9.7, 19.8, 11.5, 15.2, 6.5, 19, 11.9, 12.7,
        17, 7.5, 5.9, 0.2, 10.5, 4.2, 0.4, 4.6, 17.6, 18.4, 14.9, 17.4, 16.6, 4, 12.1, 14.3, 2.8, 18.6, 10.3
    ],
    "salary": [
        3166.9, 3126.9, 3278.8, 2828.8, 2728.7, 2762.6, 2142.6, 3214.5, 1518.9, 1049.7, 1867.9, 2390.7, 3405.8, 2449.8,
        3158.5, 1212.5, 3257.5, 3217, 1692.7, 2671.8, 3191.9, 3119.9, 2184.8, 2814, 3174.2, 1761, 2845.1, 3086.5, 1566.7,
        3244.4, 1570.7, 3052.7, 3152.7, 3316.9, 3073.8, 1269.4, 3215, 3350.7, 2499.2, 1763.9, 2813.5, 2671.1, 1867.9,
        2927.9, 2376.4, 1863, 3267.1, 2271.2, 3078.1, 1527.6, 2165, 1330.4, 1943.8, 2580.9, 1411.8, 3253.4, 1506.9,
        3288.4, 2673.8, 3212.5, 3030.1, 2976.5, 3185.7, 2821.3, 3042.4, 3328.9, 2878.2, 3270.2, 3159.8, 3064.9, 3042,
        2815.9, 3175.4, 3003.1, 3020.3, 3097.6, 3332, 2520.1, 3051.6, 3252.6, 3078.7, 3130.4, 2487.7, 2385.2, 1046.4,
        2980.3, 2070.2, 1184.3, 2035.6, 3216.5, 3232.4, 2988, 3285.2, 3373.3, 2195.4, 3119.2, 3399.5, 1596, 3051.2, 2990.9
    ]
}

X = np.array(data['experience_years']).reshape(-1, 1)
y = np.array(data['salary'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on test data
y_pred = dt_model.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, dt_model.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)
train_r2 = r2_score(y_train, dt_model.predict(X_train))
test_r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: Train = {train_mse:.2f}, Test = {test_mse:.2f}")
print(f"R^2 Score: Train = {train_r2:.2f}, Test = {test_r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', s=50)
plt.scatter(X_test, y_pred, color='red', label='Predicted', s=50)
plt.title('Decision Tree Regression: Test Data vs Test Prediction')
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()
