import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os


df = pd.read_csv("test_data.csv", sep=",")

# check the base information of this dataset
print "There are", len(df.columns), "columns:"
for x in df.columns:
    sys.stdout.write(str(x)+", ")

print df.info()

print df.head()

# using the heat map to show the correlation between these attributes
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()





