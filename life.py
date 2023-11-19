import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import trim_mean, iqr, pearsonr, stats
import statsmodels.api as sm

life = pd.read_csv('/Users/gabecrain/Desktop/GDP_Project/life_expectancy.csv')
print('column names:', list(life.columns))

life = life.rename(columns={'Life expectancy at birth (years)': 'life_expectancy', 'Country': 'country', 'Year': 'year', 'GDP': 'gdp'})

#practice using time series data
print('countries in the dataset:', life.country.unique())
print(life.head())

# chile_years = life.year[life.country == 'Chile']
# chile_life_expectancy = life.life_expectancy[life.country == 'Chile']

# plt.plot(chile_years, chile_life_expectancy)
# plt.xlabel('Year')
# plt.ylabel('Life Expectancy')
# plt.title('Chile Life Expectancy by Year')
# plt.show()
# plt.clf()

# china_years = life.year[life.country == 'China']
# china_life_expectancy = life.life_expectancy[life.country == 'China']

# plt.plot(china_years, china_life_expectancy)
# plt.xlabel('Year')
# plt.ylabel('Life Expectancy')
# plt.title('China Life Expectancy by Year')
# plt.show()
# plt.clf()

germany_years = life.year[life.country == 'Germany']
germany_life_expectancy = life.life_expectancy[life.country == 'Germany']

plt.plot(germany_years, germany_life_expectancy)
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title('Germany Life Expectancy by Year')
plt.show()
plt.clf()