import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "employee_id": [
        "EM_101", "EM_102", "EM_103", "EM_104", "EM_105", "EM_106", "EM_107", "EM_108", "EM_109", "EM_110", 
        "EM_111", "EM_112", "EM_113", "EM_114", "EM_115", "EM_116", "EM_117", "EM_118", "EM_119", "EM_120",
        "EM_121", "EM_122", "EM_123", "EM_124", "EM_125", "EM_126", "EM_127", "EM_128", "EM_129", "EM_130",
        "EM_131", "EM_132", "EM_133", "EM_134", "EM_135", "EM_136", "EM_137", "EM_138", "EM_139", "EM_140",
        "EM_141", "EM_142", "EM_144", "EM_145", "EM_146", "EM_147", "EM_148", "EM_149", "EM_150", "EM_151",
        "EM_152", "EM_153", "EM_154", "EM_155", "EM_156", "EM_157", "EM_158", "EM_159", "EM_160", "EM_161",
        "EM_162", "EM_163", "EM_164", "EM_165", "EM_166", "EM_167", "EM_168", "EM_169", "EM_170", "EM_171",
        "EM_172", "EM_173", "EM_174", "EM_175", "EM_176", "EM_177", "EM_178", "EM_179", "EM_180", "EM_181",
        "EM_182", "EM_183", "EM_184", "EM_185", "EM_186", "EM_187", "EM_188", "EM_189", "EM_190", "EM_191",
        "EM_192", "EM_193", "EM_194", "EM_195", "EM_196", "EM_197", "EM_198", "EM_199", "EM_200"
    ],
    "experience_years": [
        16.8, 10.7, 14.1, 9.1, 8.9, 7.9, 4.4, 16.2, 2, 0, 3.6, 6.1, 14.7, 6.7, 18.2, 2.8, 15.4, 15.6, 2.4, 6.3,
        11.1, 17.8, 5.3, 8.5, 12.9, 3, 8.1, 19.4, 1.8, 14.5, 2.2, 9.5, 17.2, 5.7, 18.8, 1, 13.3, 13.1, 7.1, 3.2,
        9.9, 6.9, 19.2, 5.5, 3.8, 16.4, 5.1, 12.5, 2.6, 4.8, 0.6, 3.4, 7.7, 1.6, 15.8, 1.2, 12.9, 7.3, 18, 11.3,
        20, 10.9, 8.7, 10.1, 13.5, 9.3, 13.7, 12.3, 19.6, 16, 8.3, 11.7, 9.7, 19.8, 11.5, 15.2, 6.5, 19, 11.9,
        12.7, 17, 7.5, 5.9, 0.2, 10.5, 4.2, 0.4, 4.6, 17.6, 18.4, 14.9, 17.4, 16.6, 4, 12.1, 14.3, 2.8, 18.6, 10.3
    ],
    "salary": [
        3166.9, 3126.9, 3278.8, 2828.8, 2728.7, 2762.6, 2142.6, 3214.5, 1518.9, 1049.7, 1867.9, 2390.7, 3405.8, 2449.8, 
        3158.5, 1212.5, 3257.5, 3217, 1692.7, 2671.8, 3191.9, 3119.9, 2184.8, 2814, 3174.2, 1761, 2845.1, 3086.5, 1566.7, 
        3244.4, 1570.7, 3052.7, 3152.7, 3316.9, 3073.8, 1269.4, 3215, 3350.7, 2499.2, 1763.9, 2813.5, 2671.1, 2927.9, 
        2376.4, 1863, 3267.1, 2271.2, 3078.1, 1527.6, 2165, 1330.4, 1943.8, 2580.9, 1411.8, 3253.4, 1506.9, 3288.4, 
        2673.8, 3212.5, 3030.1, 2976.5, 3185.7, 2821.3, 3042.4, 3328.9, 2878.2, 3270.2, 3159.8, 3064.9, 3042, 2815.9, 
        3175.4, 3003.1, 3020.3, 3097.6, 3332, 2520.1, 3051.6, 3252.6, 3078.7, 3130.4, 2487.7, 2385.2, 1046.4, 2980.3, 
        2070.2, 1184.3, 2035.6, 3216.5, 3232.4, 2988, 3285.2, 3373.3, 2195.4, 3119.2, 3399.5, 1596, 3051.2, 2990.9
    ]
}


X = np.array(data["experience_years"]).reshape(-1, 1)
y = np.array(data["salary"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)

# Evaluation
mse_train = mean_squared_error(y_train, lr_model.predict(X_train))
mse_test = mean_squared_error(y_test, y_pred)
r2_train = r2_score(y_train, lr_model.predict(X_train))
r2_test = r2_score(y_test, y_pred)

print("Linear Regression Results:")
print(f"Mean Squared Error (Train): {mse_train}")
print(f"Mean Squared Error (Test): {mse_test}")
print(f"R^2 Score (Train): {r2_train}")
print(f"R^2 Score (Test): {r2_test}")

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.scatter(X_test, y_pred, color="red", label="Predicted", alpha=0.6)
plt.plot(X_test, y_pred, color="green", label="Regression Line")

plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid()
plt.show()
