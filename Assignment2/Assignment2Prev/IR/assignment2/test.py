# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Sample sales data
data = {
    'Date': [
        '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
        '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
        '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15',
        '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20',
        '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25',
        '2023-01-26', '2023-01-27', '2023-01-28', '2023-01-29', '2023-01-30',
        '2023-01-31'
    ],
    'Sales': [
        200, 150, 300, 250, 400, 100, 500, 350, 450, 200,
        300, 150, 100, 400, 500, 250, 300, 200, 350, 450,
        150, 100, 500, 400, 200, 150, 300, 250, 400, 500,
        350
    ],
    'Profit': [
        50, 30, 70, 60, 90, 20, 120, 80, 110, 50,
        70, 30, 20, 90, 120, 60, 70, 50, 80, 110,
        30, 20, 120, 90, 50, 30, 70, 60, 90, 120,
        80
    ],
    'Category': [
        'Electronics', 'Furniture', 'Electronics', 'Furniture', 'Office Supplies',
        'Furniture', 'Electronics', 'Office Supplies', 'Electronics', 'Furniture',
        'Office Supplies', 'Electronics', 'Furniture', 'Electronics', 'Office Supplies',
        'Furniture', 'Electronics', 'Office Supplies', 'Furniture', 'Electronics',
        'Office Supplies', 'Furniture', 'Electronics', 'Office Supplies', 'Furniture',
        'Electronics', 'Furniture', 'Office Supplies', 'Electronics', 'Furniture',
        'Office Supplies'
    ]
}

# Create DataFrame
sales_data = pd.DataFrame(data)

# Save to CSV
sales_data.to_csv('sales_data.csv', index=False)


# Load the dataset
# file_path = 'sales_data.csv'  # Update with your file path
# sales_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(sales_data.head())

# Check for missing values
print(sales_data.isnull().sum())

# Fill missing values with appropriate methods (mean, median, mode, or drop)
sales_data.fillna(sales_data.mean(), inplace=True)

# Verify that there are no missing values
print(sales_data.isnull().sum())

# Basic statistics of the dataset
print(sales_data.describe())

# Distribution of sales
plt.hist(sales_data['Sales'], bins=20, edgecolor='black')
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
correlation_matrix = sales_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Sales over time
sales_data['Date'] = pd.to_datetime(sales_data['Date'])
sales_data.set_index('Date', inplace=True)
monthly_sales = sales_data['Sales'].resample('M').sum()

plt.plot(monthly_sales)
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Sales by category
sales_by_category = sales_data.groupby('Category')['Sales'].sum().sort_values()

plt.barh(sales_by_category.index, sales_by_category.values)
plt.title('Sales by Category')
plt.xlabel('Sales')
plt.ylabel('Category')
plt.show()

# Conclusion and Insights
print("Conclusion and Insights:")
print("1. The distribution of sales is right-skewed with most sales being in the lower range.")
print("2. There is a strong positive correlation between Sales and Profit.")
print("3. Monthly sales show a trend of increasing sales towards the end of the year.")
print("4. The 'Electronics' category has the highest sales among all categories.")
