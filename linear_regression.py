import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing

file = 'data/IRIS.csv'
df = pd.read_csv(file)
test_data = pd.read_csv('data/IRIS-answer.csv')

features = ['petal_length']
target = ['species']

encoder = preprocessing.LabelEncoder()
encoder.fit(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'])
df['species'] = encoder.transform(df['species']);
test_data['species'] = encoder.transform(test_data['species'])

x_train = df[features].dropna()
y_train = df[target].dropna()
x_test = test_data[features].dropna()
y_test = test_data[target].dropna()

l_model = linear_model.LinearRegression()
l_model.fit(x_train, y_train)

score_train = l_model.score(x_train, y_train)
score_test = l_model.score(x_test, y_test)
print(score_train, score_test)
