#!/usr/bin/env python
# coding: utf-8

# # Improving Efficiency for Electric Vehicles
# 
# 

# ## 1. Define the Problem
# Electric vehicles (EVs) are becoming popular due to sustainability, but efficiency depends on factors like battery capacity, range, and energy consumption.
# 
# **Problem Statement:** How can we analyze EV data to understand factors affecting range and efficiency, and prepare the dataset for predictive modeling?

# ## 2. Data Collection & Understanding

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.DataFrame({
    'Make': ['Tesla', 'Nissan', 'BMW', 'Hyundai', 'Tata', 'MG', 'Kia', 'Audi'],
    'Model': ['Model 3', 'Leaf', 'i3', 'Kona', 'Nexon', 'ZS EV', 'EV6', 'e-tron'],
    'Battery_Capacity_kWh': [60.0, 40.0, 42.0, 64.0, 30.2, 44.5, 77.4, 95.0],
    'Range_km': [450, 240, 260, 452, 312, 335, 528, 436],
    'Energy_Consumption_WhPerKm': [133, 166, 161, 141, 193, 155, 146, 218],
    'Fast_Charging': [1, 0, 1, 1, 1, 1, 1, 1]
})
data.head()


# ## 3. Data Preprocessing

# In[2]:


print(data.info())
print("\nMissing values:\n", data.isnull().sum())

# Encode categorical variables
data_encoded = pd.get_dummies(data, columns=['Make', 'Model'], drop_first=True)
data_encoded.head()


# ## 4. Data Splitting

# In[3]:


from sklearn.model_selection import train_test_split

X = data_encoded.drop('Range_km', axis=1)
y = data_encoded['Range_km']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape


# ## 5. Exploratory Data Analysis (EDA)

# In[4]:


sns.pairplot(data[['Battery_Capacity_kWh','Range_km','Energy_Consumption_WhPerKm']])
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(data[['Battery_Capacity_kWh','Range_km','Energy_Consumption_WhPerKm']].corr(), annot=True, cmap='coolwarm')
plt.show()


# ## 6. Feature Engineering

# In[5]:


# Create new feature: efficiency (km per kWh)
data['Efficiency_kmPerkWh'] = data['Range_km'] / data['Battery_Capacity_kWh']
data[['Model','Efficiency_kmPerkWh']]


# ## 7. Advanced Visualization

# In[6]:


plt.figure(figsize=(10,6))
sns.barplot(x='Make', y='Efficiency_kmPerkWh', data=data)
plt.title('Efficiency Comparison by Brand')
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='Battery_Capacity_kWh', y='Range_km', hue='Make', data=data, s=100)
plt.title('Range vs Battery Capacity')
plt.show()


# ## 8. Modeling Preparation

# In[7]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## 9. Model Training and Evaluation

# In[8]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# ## 10. Prediction Example

# In[9]:


sample = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample)
predicted_range = model.predict(sample_scaled)
print("Predicted Range (km):", predicted_range[0])
print("Actual Range (km):", y_test.iloc[0])


# ## 11. Efficiency Analysis

# In[10]:


plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='Efficiency_kmPerkWh', data=data)
plt.xticks(rotation=45)
plt.title('EV Efficiency Analysis (km per kWh)')
plt.show()

best_efficiency = data.sort_values(by='Efficiency_kmPerkWh', ascending=False).head(3)
print("Top Efficient EVs:\n", best_efficiency[['Make','Model','Efficiency_kmPerkWh']])

