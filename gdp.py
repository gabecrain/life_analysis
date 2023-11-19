import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import trim_mean, iqr, pearsonr, stats
import statsmodels.api as sm
import plotly.express as px

life = pd.read_csv('/Users/gabecrain/Desktop/GDP_Project/life_expectancy.csv')
print('column names:', list(life.columns))

life = life.rename(columns={'Life expectancy at birth (years)': 'life_expectancy', 'Country': 'country', 'Year': 'year', 'GDP': 'gdp'})

#practice using time series data
print('countries in the dataset:', life.country.unique())
print(life.head())

mexico_years = life.year[life.country == 'Mexico']
usa_years = life.year[life.country == 'United States of America']
chile_years = life.year[life.country == 'Chile']
germany_years = life.year[life.country == 'Germany']
china_years = life.year[life.country == 'China']
zimbabwe_years = life.year[life.country == 'Zimbabwe']

mexico_life_expectancy = life.life_expectancy[life.country == 'Mexico']
usa_life_expectancy = life.life_expectancy[life.country == 'United States of America']
chile_life_expectancy = life.life_expectancy[life.country == 'Chile']
china_life_expectancy = life.life_expectancy[life.country == 'China']
germany_life_expectancy = life.life_expectancy[life.country == 'Germany']
zimbabwe_life_expectancy = life.life_expectancy[life.country == 'Zimbabwe']

mexico_gdp = life.gdp[life.country == 'Mexico']
usa_gdp = life.gdp[life.country == 'United States of America']
chile_gdp = life.gdp[life.country == 'Chile']
china_gdp= life.gdp[life.country == 'China']
germany_gdp = life.gdp[life.country == 'Germany']
zimbabwe_gdp = life.gdp[life.country == 'Zimbabwe']

#plot all country life expectancy by year
plt.plot(usa_years, usa_life_expectancy, marker='o', label='USA')
plt.plot(mexico_years, mexico_life_expectancy, marker='o', label='Mexico')
plt.plot(chile_years, chile_life_expectancy, marker='o', label='Chile')
plt.plot(china_years, china_life_expectancy, marker='o', label='China')
plt.plot(germany_years, germany_life_expectancy, marker='o', label='Germany')
plt.plot(zimbabwe_years, zimbabwe_life_expectancy, marker='o', label='Zimbabwe')
plt.legend()
plt.title('Country Life Expectancy by Year')
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.savefig('country_life_expectancy.png')
plt.show()
plt.clf()

#plot all country gdp by year
plt.plot(usa_years, usa_gdp, marker='o', label='USA')
plt.plot(mexico_years, mexico_gdp, marker='o', label='Mexico')
plt.plot(chile_years, chile_gdp, marker='o', label='Chile')
plt.plot(china_years, china_gdp, marker='o', label='China')
plt.plot(germany_years, germany_gdp, marker='o', label='Germany')
plt.plot(zimbabwe_years, zimbabwe_gdp, marker='o', label='Zimbabwe')
plt.legend()
plt.title('Country GDP by Year')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.savefig('country_gdp.png')
# plt.show()
plt.clf()

plt.plot(chile_years, chile_life_expectancy)
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title('Chile Life Expectancy by Year')
# plt.show()
plt.clf()

plt.plot(china_years, china_life_expectancy)
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title('China Life Expectancy by Year')
# plt.show()
plt.clf()

plt.plot(germany_years, germany_life_expectancy)
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title('Germany Life Expectancy by Year')
# plt.show()
plt.clf()

plt.plot(germany_years, germany_gdp)
plt.xlabel('Year')
plt.ylabel('GDP (USD)')
plt.title('Germany GDP by Year')
# plt.show()
plt.clf()

#use linear regression to explore relationship between life expectancy and gdp for Zimbabwe
#fit linear regression model
zimbabwe_life = life[life.country == 'Zimbabwe']
zimbabwe_model = sm.OLS.from_formula('life_expectancy ~ gdp', data=zimbabwe_life)
zimbabwe_model_results = zimbabwe_model.fit()
print('Results of Zimbabwe model:\n', zimbabwe_model_results.params)
#what can we predict the life expectancy to be of zimbabwe if they had a gdp of 1.8e10?
print('predicted life expectancy of Zimbabwe with GDP of 1.8^10:', zimbabwe_model_results.params[1]*1.8**10 + zimbabwe_model_results.params[0])

#plot Zimbabwe life expectancy and gdp over regression line
plt.scatter(zimbabwe_life.gdp, zimbabwe_life.life_expectancy)
plt.plot(zimbabwe_life.gdp, zimbabwe_model_results.predict(zimbabwe_life))
plt.xlabel('GDP (USD)')
plt.ylabel('Life Expectancy')
plt.title('Zimbabwe Life Expectancy and GDP')
# plt.show()
plt.clf()

#verify normality and homoscedasticity for the model
zimbabwe_regression_fitted = zimbabwe_model_results.predict(zimbabwe_life)
print('fitted values of zimbabwe regression model:\n', zimbabwe_regression_fitted.head())
zimbabwe_regression_residuals = zimbabwe_life.life_expectancy - zimbabwe_regression_fitted
print('residuals of zimbabwe regression model:\n', zimbabwe_regression_residuals.head())

#plot histogram of zimbabwe model residuals
plt.hist(zimbabwe_regression_residuals)
# plt.show()
plt.clf()
#plot does not appear to be normally distributed, indicating that the assumption is not satisfied.

plt.scatter(zimbabwe_regression_fitted, zimbabwe_regression_residuals)
# plt.show()
plt.clf()
#it does not appear that the residuals have equal variation across all values, though there are so few data points it is difficult to tell.


#compare life expectancy across all countries for the year 2000
life_expectancy_2000 = life[life['year'] == 2000]
sns.barplot(x='country', y='life_expectancy', data=life_expectancy_2000)
plt.title('Life Expectancy by Country in 2000')
plt.xlabel('Country')
plt.ylabel('Life Expectancy')
plt.xticks(rotation=-45)
# plt.show()
plt.clf()
#all countries for the year 2000 have a similar life expectancy near 70 with the exception of Zimbabwe which is near 40.


#explore relationship between life expectancy and GDP by country using scatterplot and boxplot
sns.scatterplot(x='gdp', y='life_expectancy', hue='country', palette='bright', data=life)
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')
plt.title('Life Expectancy and GDP by Country')
# plt.show()
plt.clf()
#scatterplot reveals several insights. China and USA had a dramatic increase in GDP whereas the other countries did not.
#all countries had an increase in life expectancy, though Zimbabwe had the most dramatic increase of around 20 years.
#the GDP of both Chile and Zimbabwe had almost no increase in GDP whatsoever for a 20 year period.

#try a 3d scatterplot to visualize all variables simultaneously
life_3d = px.scatter_3d(life, x='gdp', y='life_expectancy', z='year', color='country')
# life_3d.show()

sns.boxplot(x='country', y='life_expectancy', palette='deep', data=life)
plt.xlabel('Country')
plt.ylabel('Life Expectancy')
plt.title('Country Life Expectancy by Year')
plt.xticks(rotation=-30)
# plt.show()
plt.clf()

sns.boxplot(x='country', y='gdp', palette='deep', data=life)
plt.xlabel('Country')
plt.ylabel('GDP')
plt.title('Country GDP by Year')
plt.xticks(rotation=-30)
# plt.show()
plt.clf()
#variance of USA and China indicate large changes in GDP, wheras other countries have very little change.

#conclusions
#analyzing this dataset lead to some insights about the relationship between GDP and life expectancy.
#there is a positive correlation between GDP and life expectancy.
#Zimbabwe and Chili had very little GDP growth over the length of the data collection. The USA and China had explosive GDP growth.
#Zimbabwe had the most dramatic improvement in life expectancy, with an increase of about 20 years.
#Both life expectancy and GDP increased for all countries during the data collection, albiet with different rates.