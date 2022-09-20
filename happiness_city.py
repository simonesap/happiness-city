import pandas as pd
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

report = read_csv('./2019.csv')
print(report.columns.values)

report_df = pd.DataFrame(report).head(50)

# GDP per capita
report_df.plot(x ='Country or region', y='GDP per capita', kind = 'bar')
plt.show()

# Healthy life expectancy
report_df.plot(x ='Country or region', y='Healthy life expectancy', kind = 'bar')
plt.show()

# Generosity
report_df.plot(x ='Country or region', y='Generosity', kind = 'bar')
plt.show()

# Perceptions of corruption
report_df.plot(x ='Country or region', y='Perceptions of corruption', kind = 'bar')
plt.show()

X = report.drop(columns=['Overall rank','Score','GDP per capita','Social support','Freedom to make life choices','Freedom to make life choices','Country or region'])
y = report['Country or region']

print(X)
print(y)

model = DecisionTreeClassifier()
model = model.fit(X.values, y.values)
# Insert inputs: Healthy life expectancy - Generosity - Perceptions of corruption
prevision = model.predict([[1.0,0.500,0.0]])
prevision_prob = model.predict_proba([[1.0,0.400,0.100]])

print('The best Country for Healthy life expectancy - Generosity - Perceptions of corruption is:', prevision)