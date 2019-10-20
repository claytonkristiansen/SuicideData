# The 'as' keyword just renames the library, making it easier for us to access it
import pandas as pd
import numpy as np
# seed the random generator to get consistent results
np.random.seed(0)
import seaborn as sns
from matplotlib import pyplot as plt
# Our score function. More explained later.
from sklearn.metrics import r2_score

from pycosmos import CosmosProject
tamu_datathon = CosmosProject('tamu_datathon')

# @title Import some libraries we need for interactive plotting
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from mpl_toolkits import mplot3d

from sklearn.linear_model import LinearRegression


suicide_data = pd.read_csv("C:/Users/Clayt/repos/TAMUDatathon/SuicideData/who_suicide_statistics.csv", sep=",")


suicide_data.head()

# This will give us descriptive statistics (such as count, mean, min/max, 
# standard deviation) for each column in out dataset
descriptive_stats = suicide_data.describe()

suicide_data_missing_to_mean = suicide_data.fillna(suicide_data.mean())

suicide_data = suicide_data_missing_to_mean

print(descriptive_stats)

x = suicide_data[["population"]]
y = suicide_data[["suicides_per_thousand"]]
suicide_model = LinearRegression().fit(x, y)
y_hat = suicide_model.predict(x)

plt.scatter(x, y, label='Suicide Data')
#plt.plot(x, suicide_model, color="r", label='Model: $\hat{{y}}$={:.2f}*x+{:.0f}'.format(suicide_model.coef_, suicide_model.intercept_))
print(suicide_model.coef_)
# Set location of the legend of the plot.
plt.legend(loc='upper left')
plt.title("Computer Fit Model ($R^2$: {:.2f})".format(r2_score(y, y_hat)))
plt.xlabel('population')
plt.ylabel('suicides per 1000')
plt.show() 
i = 0