import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the Excel file
import openpyxl
print("openpyxl installed successfully!")

file_path = r"C:\Users\jamih\Desktop\Aged-Aspect Ratio_N.xlsx"
xls = pd.ExcelFile(file_path)

# Initialize a dictionary to store data from each sheet
data_dict = {}

# Step 2: Iterate through each sheet and calculate aspect ratio
for sheet_name in xls.sheet_names:
    df = xls.parse(sheet_name)
    df['Aspect Ratio'] = df['Diameter (nm)'] / df['Length (nm)']
    data_dict[sheet_name] = df

# Step 3: Calculate chirality index (standard deviation of aspect ratios)
chirality_index = {}
for sheet_name, df in data_dict.items():
    chirality_index[sheet_name] = df['Aspect Ratio'].std()

# Print chirality index for each sheet
print("Chirality Index for Each Sheet:")
for sheet_name, ci in chirality_index.items():
    print(f"Sheet: {sheet_name}, Chirality Index: {ci}")

# Step 4: Compare crystallography by plotting chirality index
# Extract crystallography information from sheet names
crystallography = [name.split('-')[1] for name in chirality_index.keys()]
ci_values = list(chirality_index.values())

# Plot chirality index vs crystallography
plt.figure(figsize=(10, 6))
plt.bar(crystallography, ci_values, color='blue')
plt.xlabel('Crystallography Orientation')
plt.ylabel('Chirality Index')
plt.title('Chirality Index vs Crystallography Orientation')
plt.show()

# Step 5: Save the results to a new Excel file
with pd.ExcelWriter(r"C:\Users\jamih\Desktop\Aged-Aspect Ratio_N_Results.xlsx") as writer:
    for sheet_name, df in data_dict.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Results saved to 'Aged-Aspect Ratio_N_Results.xlsx'.")

