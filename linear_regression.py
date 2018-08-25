import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing

file = 'data/IRIS.csv'
df = pd.read_csv(file)

features = ['petal_length']
target = ['species']

encoder = preprocessing.LabelEncoder()
encoder.fit(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'])
df['species'] = encoder.transform(df['species']);

train, test = train_test_split(df, test_size=0.30)
x_train = train[features].dropna()
y_train = train[target].dropna()
x_test = test[features].dropna()
y_test = test[target].dropna()

l_model = linear_model.LinearRegression()
l_model.fit(x_train, y_train)

score_train = l_model.score(x_train, y_train)
score_test = l_model.score(x_test, y_test)
print(score_train, score_test)
