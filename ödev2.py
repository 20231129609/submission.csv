import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Veri setlerini yükleme
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Basit veri ön işleme
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

# Özellik seçimi
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_train = train[features]
y_train = train['Survived']
X_test = test[features]

# Model oluşturma ve eğitim
model1 = RandomForestClassifier(random_state=42)
model2 = XGBClassifier(random_state=42)
model3 = LogisticRegression(random_state=42)
ensemble = VotingClassifier(estimators=[('rf', model1), ('xgb', model2), ('lr', model3)], voting='hard')
ensemble.fit(X_train, y_train)

# Tahmin
predictions = ensemble.predict(X_test)

# Gönderi dosyası oluşturma
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
submission.to_csv('submission_ensemble.csv', index=False)
print("Tahminler 'submission_ensemble.csv' dosyasına kaydedildi!")