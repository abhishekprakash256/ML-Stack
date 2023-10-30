import pandas as pd

# Sample DataFrame
data = {'Text': ['apple', 'banana', 'cherry', 'date', 'elderberry']}
df = pd.DataFrame(data)

# Applying multiple OR conditions to filter rows
#filter1 = df['Text'].str.contains('a')  # Rows containing 'a'
filter2 = df['Text'].str.contains('n')  # Rows containing 'e'
filter3 = df['Text'].str.contains('y')  # Rows containing 'y'

# Combine the conditions using the | (pipe) operator
filtered_df = df[ filter2 | filter3]

print(filtered_df["Text"])
