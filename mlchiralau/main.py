# pip install pandas numpy matplotlib scipy scikit-learn openpyxl
# #type this in terminal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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

# Step 5: Compare crystallography
crystallography = [name.split('-')[1] for name in chirality_index.keys()]
ci_values = list(chirality_index.values())
cp_values = list(chirality_parameter.values())

plt.figure(figsize=(12, 6))
plt.bar(crystallography, ci_values, color='blue', alpha=0.6, label='Chirality Index')
plt.bar(crystallography, cp_values, color='green', alpha=0.6, label='Chirality Parameter')
plt.xlabel('Crystallography Orientation')
plt.ylabel('Chirality Value')
plt.title('Chirality Metrics vs Crystallography Orientation')
plt.legend()
plt.show()

# Step 6: Prepare data for machine learning
# Create a DataFrame with features (Aspect Ratio, Chirality Index, Chirality Parameter) and labels (Crystallography)
ml_data = []
for sheet_name, df in data_dict.items():
    crystallography_label = sheet_name.split('-')[1]
    for _, row in df.iterrows():
        ml_data.append([
            row['Aspect Ratio'],
            chirality_index[sheet_name],
            chirality_parameter[sheet_name],
            crystallography_label
        ])

ml_df = pd.DataFrame(ml_data, columns=['Aspect Ratio', 'Chirality Index', 'Chirality Parameter', 'Crystallography'])

# Step 7: Train a machine learning model
# Features (X) and labels (y)
X = ml_df[['Aspect Ratio', 'Chirality Index', 'Chirality Parameter']]
y = ml_df['Crystallography']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Step 8: Save all results to a new Excel file
with pd.ExcelWriter(r"C:\Users\jamih\Desktop\Chirality_Analysis_Results.xlsx") as writer:
    for sheet_name, df in data_dict.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    ml_df.to_excel(writer, sheet_name='ML_Data', index=False)
print("Results saved to 'Chirality_Analysis_Results.xlsx'.")