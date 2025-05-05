import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Veri setlerini yükleme
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# İsimlerden unvan çıkarma
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Unvanları gruplandırma
train['Title'] = train['Title'].replace(['Lady', 'Countess', 'Sir', 'Jonkheer', 'Don', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace(['Mlle', 'Ms'], 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
test['Title'] = test['Title'].replace(['Lady', 'Countess', 'Sir', 'Jonkheer', 'Don', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace(['Mlle', 'Ms'], 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')

# Eksik verileri doldurma
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Kategorik verileri dönüştürme
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])
train['Embarked'] = le.fit_transform(train['Embarked'])
test['Embarked'] = le.transform(test['Embarked'])
train['Title'] = le.fit_transform(train['Title'])
test['Title'] = le.transform(test['Title'])

# Özellik seçimi (Title özelliği eklendi)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']
X_train = train[features]
y_train = train['Survived']
X_test = test[features]

# Model eğitimi
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin
predictions = model.predict(X_test)

# Gönderi dosyası oluşturma
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
submission.to_csv('submission_with_titles.csv', index=False)
print("Tahminler 'submission_with_titles.csv' dosyasına kaydedildi!")

# Özellik önemliliklerini yazdırma
feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
print("\nÖzellik önemlilikleri:")
print(feature_importance.sort_values('importance', ascending=False))