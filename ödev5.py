# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Grafik stilini ayarlama
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Veri setlerini yükleme
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# --- Veri Görselleştirme ---
# 1. Hayatta Kalma Oranları (Genel)
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=train)
plt.title('Hayatta Kalma Dağılımı (0: Hayatta Kalmadı, 1: Hayatta Kaldı)')
plt.xlabel('Hayatta Kalma')
plt.ylabel('Yolcu Sayısı')
plt.savefig('hayatta_kalma_dagilimi.png')
plt.close()

# 2. Cinsiyete Göre Hayatta Kalma Oranı
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=train)
plt.title('Cinsiyete Göre Hayatta Kalma Oranı')
plt.xlabel('Cinsiyet')
plt.ylabel('Hayatta Kalma Oranı')
plt.savefig('cinsiyet_hayatta_kalma.png')
plt.close()

# 3. Bilet Sınıfına Göre Hayatta Kalma Oranı
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train)
plt.title('Bilet Sınıfına ve Cinsiyete Göre Hayatta Kalma Oranı')
plt.xlabel('Bilet Sınıfı')
plt.ylabel('Hayatta Kalma Oranı')
plt.savefig('sinif_cinsiyet_hayatta_kalma.png')
plt.close()

# 4. Yaş Dağılımı
plt.figure(figsize=(10, 5))
sns.histplot(train['Age'].dropna(), bins=30, kde=True)
plt.title('Yaş Dağılımı')
plt.xlabel('Yaş')
plt.ylabel('Frekans')
plt.savefig('yas_dagilimi.png')
plt.close()

# 5. Korelasyon Matrisi
plt.figure(figsize=(10, 8))
numeric_cols = train.select_dtypes(include=[np.number]).columns
corr = train[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Özellikler Arası Korelasyon Matrisi')
plt.savefig('korelasyon_matrisi.png')
plt.close()

# 6. Bilet Ücreti ve Sınıf İlişkisi (Kutu Grafiği)
plt.figure(figsize=(10, 5))
sns.boxplot(x='Pclass', y='Fare', data=train)
plt.title('Bilet Sınıfına Göre Bilet Ücreti Dağılımı')
plt.xlabel('Bilet Sınıfı')
plt.ylabel('Bilet Ücreti')
plt.savefig('bilet_ucreti_dagilimi.png')
plt.close()

# --- Özellik Mühendisliği ---
# Unvan çıkarma
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace(['Lady', 'Countess', 'Sir', 'Jonkheer', 'Don', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace(['Lady', 'Countess', 'Sir', 'Jonkheer', 'Don', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
test['Title'] = test['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')

# Unvanlara Göre Hayatta Kalma Oranı (Görselleştirme)
plt.figure(figsize=(10, 5))
sns.barplot(x='Title', y='Survived', data=train)
plt.title('Unvana Göre Hayatta Kalma Oranı')
plt.xlabel('Unvan')
plt.ylabel('Hayatta Kalma Oranı')
plt.savefig('unvan_hayatta_kalma.png')
plt.close()

# Aile büyüklüğü
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

# Aile Büyüklüğüne Göre Hayatta Kalma Oranı (Görselleştirme)
plt.figure(figsize=(10, 5))
sns.barplot(x='FamilySize', y='Survived', data=train)
plt.title('Aile Büyüklüğüne Göre Hayatta Kalma Oranı')
plt.xlabel('Aile Büyüklüğü')
plt.ylabel('Hayatta Kalma Oranı')
plt.savefig('aile_buyuklugu_hayatta_kalma.png')
plt.close()

# --- Eksik Verileri Doldurma ---
train['Age'] = train.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
test['Age'] = test.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# --- Kategorik Verileri Dönüştürme ---
le = LabelEncoder()
for col in ['Sex', 'Embarked', 'Title']:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# --- Özellik Seçimi ---
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
X_train = train[features]
y_train = train['Survived']
X_test = test[features]

# --- Model Eğitimi ---
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

# Çapraz Doğrulama
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Çapraz Doğrulama Skorları:", scores)
print("Ortalama Doğruluk:", scores.mean())

# Modeli eğitme
model.fit(X_train, y_train)

# Özellik Önem Dereceleri (Görselleştirme)
plt.figure(figsize=(10, 5))
feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Özellik Önem Dereceleri')
plt.xlabel('Önem Skoru')
plt.ylabel('Özellik')
plt.savefig('ozellik_onem_dereceleri.png')
plt.close()

# --- Tahmin ---
predictions = model.predict(X_test)

# --- Gönderi Dosyası Oluşturma ---
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
submission.to_csv('submission.csv', index=False)
print("Tahminler 'submission.csv' dosyasına kaydedildi!")