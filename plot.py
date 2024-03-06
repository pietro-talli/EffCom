import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('results/1/1_results.csv')

# Extract columns for the first set of points
x1 = df.iloc[:, 4]  # 5th column
y1 = df.iloc[:, 3]  # 4th column

# Extract columns for the second set of points
x2 = df.iloc[:, 6]  # 7th column
y2 = df.iloc[:, 5]  # 6th column

# Plot the first set of points
plt.scatter(x1, y1, color='blue', label='First Set')

# Plot the second set of points
plt.scatter(x2, y2, color='red', label='Second Set')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot of Two Sets of Points')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
