
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1: Load the dataset (CSV format)
file_path = "Aged_Aspect_Ratio.csv"  #Uploaded to GitHub

if not os.path.exists(file_path):
    print(f"Error: Data file '{file_path}' not found. Please upload it to the repository.")
    exit()

df = pd.read_csv(file_path)

# Step 2: Calculate aspect ratio (Diameter / Length)
df['Aspect Ratio'] = df['Diameter (nm)'] / df['Length (nm)']

# Step 3: Calculate chirality index (standard deviation of aspect ratios)
chirality_index = df.groupby('Crystallography')['Aspect Ratio'].std()

# Print chirality index
print("Chirality Index for Each Crystallography Orientation:")
print(chirality_index)

# Step 4: Compare crystallography by plotting chirality index
plt.figure(figsize=(10, 6))
chirality_index.plot(kind='bar', color='blue')
plt.xlabel('Crystallography Orientation')
plt.ylabel('Chirality Index')
plt.title('Chirality Index vs Crystallography Orientation')
plt.show()

# Step 5: Save the results to a new CSV file
df.to_csv("Aged_Aspect_Ratio_Results.csv", index=False)
print("Results saved to 'Aged_Aspect_Ratio_Results.csv'.")
=======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1: Load the dataset (CSV format)
file_path = "Aged_Aspect_Ratio.csv"  #Uploaded to GitHub

if not os.path.exists(file_path):
    print(f"Error: Data file '{file_path}' not found. Please upload it to the repository.")
    exit()

df = pd.read_csv(file_path)

# Step 2: Calculate aspect ratio (Diameter / Length)
df['Aspect Ratio'] = df['Diameter (nm)'] / df['Length (nm)']

# Step 3: Calculate chirality index (standard deviation of aspect ratios)
chirality_index = df.groupby('Crystallography')['Aspect Ratio'].std()

# Print chirality index
print("Chirality Index for Each Crystallography Orientation:")
print(chirality_index)

# Step 4: Compare crystallography by plotting chirality index
plt.figure(figsize=(10, 6))
chirality_index.plot(kind='bar', color='blue')
plt.xlabel('Crystallography Orientation')
plt.ylabel('Chirality Index')
plt.title('Chirality Index vs Crystallography Orientation')
plt.show()

# Step 5: Save the results to a new CSV file
df.to_csv("Aged_Aspect_Ratio_Results.csv", index=False)
print("Results saved to 'Aged_Aspect_Ratio_Results.csv'.")

