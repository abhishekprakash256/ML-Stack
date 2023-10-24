"""
plotting the tips dataset using seaborn
"""

# Import seaborn
import seaborn as sns
import pandas as pd
import torch as th
from torch import nn
import matplotlib.pyplot as plt



FILE_PATH = "../datasets/tips_full.csv"

# Apply the default theme
sns.set_theme()

# Load an example dataset
df = pd.read_csv(FILE_PATH)

print(df.info())


# Create a visualization
sns.relplot(
    data=df,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)

plt.show()
