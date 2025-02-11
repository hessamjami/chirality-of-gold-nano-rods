import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

# Step 1: Load the Excel file
file_path = r"C:\Users\jamih\Desktop\Aged-Aspect Ratio_N.xlsx"
xls = pd.ExcelFile(file_path)

# Initialize a dictionary to store data from each sheet
data_dict = {}

# Step 2: Iterate through each sheet and calculate aspect ratio
for sheet_name in xls.sheet_names:
    df = xls.parse(sheet_name)
    df['Aspect Ratio'] = df['Diameter (nm)'] / df['Length (nm)']
    data_dict[sheet_name] = df

# Step 3: Calculate chirality index and chirality parameter
chirality_index = {}
chirality_parameter = {}
for sheet_name, df in data_dict.items():
    aspect_ratios = df['Aspect Ratio']
    chirality_index[sheet_name] = aspect_ratios.std()
    skewness = skew(aspect_ratios)
    kurt = kurtosis(aspect_ratios)
    chirality_parameter[sheet_name] = skewness / kurt if kurt != 0 else np.nan

# Step 4: Print chirality metrics
print("Chirality Metrics for Each Sheet:")
for sheet_name in data_dict.keys():
    print(f"Sheet: {sheet_name}")
    print(f"  Chirality Index: {chirality_index[sheet_name]}")
    print(f"  Chirality Parameter: {chirality_parameter[sheet_name]}")

# Step 5: Prepare data for machine learning
# Create a DataFrame with features (Aspect Ratio, Chirality Index, Chirality Parameter) and labels (Chirality or crystallography)
ml_data = []
for sheet_name, df in data_dict.items():
    for _, row in df.iterrows():
        ml_data.append([
            row['Aspect Ratio'],
            chirality_index[sheet_name],
            chirality_parameter[sheet_name]
        ])

ml_df = pd.DataFrame(ml_data, columns=['Aspect Ratio', 'Chirality Index', 'Chirality Parameter'])

# Define chirality as the target (e.g., can be binary for classification or continuous for regression)
# For simplicity, we'll assume chirality is classified as 'High' or 'Low' based on chirality index.
ml_df['Chirality'] = np.where(ml_df['Chirality Index'] > ml_df['Chirality Index'].median(), 'High', 'Low')

# Step 6: Train a machine learning model
# Features (X) and labels (y)
X = ml_df[['Aspect Ratio', 'Chirality Index', 'Chirality Parameter']]
y = ml_df['Chirality']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier (use RandomForestRegressor if predicting continuous chirality)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Step 7: Visualize results for chirality prediction
plt.figure(figsize=(12, 6))
plt.bar(ml_df['Chirality'], ml_df['Chirality Index'], color='blue', alpha=0.6, label='Chirality Index')
plt.xlabel('Chirality Class')
plt.ylabel('Chirality Index Value')
plt.title('Chirality Index vs Chirality Class')
plt.legend()
plt.show()

# Step 8: Save all results to a new Excel file
with pd.ExcelWriter(r"C:\Users\jamih\Desktop\Chirality_Analysis_Results.xlsx") as writer:
    for sheet_name, df in data_dict.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    ml_df.to_excel(writer, sheet_name='ML_Data', index=False)
print("Results saved to 'Chirality_Analysis_Results.xlsx'.")
