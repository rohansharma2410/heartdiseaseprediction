import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv('patient_appointments.csv') 


selected_features = ['age', 'gender', 'location', 'appointment_date', 'appointment_time', 'day_of_week', 'waiting_time', 'previous_appointments', 'health_conditions', 'medication', 'communication_channel', 'reminder_frequency', 'employment_status', 'income_level', 'insurance_coverage', 'no_show_history', 'booking_channel', 'prior_engagement']

# Filter the data based on the selected features
filtered_data = data[selected_features + ['attendance']].copy()

# Perform any additional preprocessing steps like encoding categorical variables, handling missing values, and scaling numeric features

# Step 2: Split the data into training and testing sets
X = filtered_data.drop('attendance', axis=1)  # Features
y = filtered_data['attendance']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions on the testing set
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 6: Make predictions on new data
# Replace 'new_data' with the relevant patient information for prediction
new_data = pd.DataFrame({
    'age': [30],
    'gender': ['Male'],
    'location': ['City'],
    'appointment_date': ['2023-07-15'],
    'appointment_time': ['10:00'],
    'day_of_week': ['Friday'],
    'waiting_time': [5],
    'previous_appointments': [3],
    'health_conditions': ['None'],
    'medication': ['No'],
    'communication_channel': ['SMS'],
    'reminder_frequency': ['Daily'],
    'employment_status': ['Employed'],
    'income_level': ['High'],
    'insurance_coverage': ['Yes'],
    'no_show_history': ['Low'],
    'booking_channel': ['Online'],
    'prior_engagement': ['No']
})

prediction = model.predict(new_data)
print("Prediction:", prediction)





import pandas as pd

# Sample data for patient appointments
data = {
    'age': [25, 32, 45, 62, 28],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'location': ['City', 'Suburb', 'City', 'Rural', 'City'],
    'appointment_date': ['2022-06-15', '2022-07-05', '2022-08-10', '2022-09-20', '2022-10-01'],
    'appointment_time': ['10:00', '14:30', '09:15', '16:45', '11:30'],
    'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Wednesday', 'Saturday'],
    'waiting_time': [10, 5, 30, 15, 7],
    'previous_appointments': [1, 0, 3, 2, 1],
    'health_conditions': ['None', 'Asthma', 'Diabetes', 'Hypertension', 'None'],
    'medication': ['No', 'Yes', 'Yes', 'Yes', 'No'],
    'communication_channel': ['SMS', 'Email', 'Phone', 'SMS', 'Phone'],
    'reminder_frequency': ['Daily', 'Weekly', 'None', 'Daily', 'None'],
    'employment_status': ['Employed', 'Unemployed', 'Employed', 'Retired', 'Employed'],
    'income_level': ['High', 'Low', 'Medium', 'Medium', 'High'],
    'insurance_coverage': ['Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'no_show_history': ['Low', 'Low', 'Medium', 'High', 'Medium'],
    'booking_channel': ['Online', 'Phone', 'Online', 'Online', 'Online'],
    'prior_engagement': ['No', 'No', 'Yes', 'No', 'No'],
    'attendance': ['Yes', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Save the DataFrame to CSV
df.to_csv('patient_appointments.csv', index=False)
