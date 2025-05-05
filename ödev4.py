import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Veri setlerini yükleme
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Özellik mühendisliği
# Unvan çıkarma
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace(['Lady', 'Countess', 'Sir', 'Jonkheer', 'Don', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace(['Lady', 'Countess', 'Sir', 'Jonkheer', 'Don', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
test['Title'] = test['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')

# Aile büyüklüğü
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + train['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

# Eksik verileri doldurma
train['Age'] = train.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
test['Age'] = test.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Kategorik verileri dönüştürme
le = LabelEncoder()
for col in ['Sex', 'Embarked', 'Title']:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Özellik seçimi
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
X_train = train[features]
y_train = train['Survived']
X_test = test[features]

# Model eğitimi
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Çapraz doğrulama skorları:", scores)
print("Ortalama doğruluk:", scores.mean())

model.fit(X_train, y_train)

# Tahmin
predictions = model.predict(X_test)

# Gönderi dosyası oluşturma
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
submission.to_csv('submission_improved.csv', index=False)
print("Tahminler 'submission_improved.csv' dosyasına kaydedildi!")