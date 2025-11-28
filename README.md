# 🏠 Ev Fiyatı Tahmin Modeli - Doğrusal Regresyon Analizi

## 📋 Proje Açıklaması

Bu Jupyter Notebook projesi, ev fiyatlarını tahmin etmek için **doğrusal regresyon** (Linear Regression) modellerini kullanan bir makine öğrenmesi çalışmasıdır. Proje, farklı regresyon algoritmalarını karşılaştırarak en iyi performansı gösteren modeli belirlemeye yöneliktir.

---

## 📊 Veri Seti

**Dosya:** `house_price_regression_dataset.csv`

**Veri Seti Özellikleri:**
- **Toplam Örnek:** 1000 ev
- **Toplam Özellik:** 7 tahmin değişkeni + 1 hedef değişken

**Özellikler (Features):**
| Özellik | Açıklama | Veri Türü |
|---------|----------|----------|
| **Square_Footage** | Evin metrekare cinsinden alanı | int64 |
| **Num_Bedrooms** | Yatak odası sayısı | int64 |
| **Num_Bathrooms** | Banyo sayısı | int64 |
| **Year_Built** | Evin yapım yılı | int64 |
| **Lot_Size** | Arsa büyüklüğü | float64 |
| **Garage_Size** | Garaj kapasitesi | int64 |
| **Neighborhood_Quality** | Mahalle kalitesi puanı | int64 |

**Hedef Değişken (Target):**
- **House_Price** 🎯 - Ev fiyatı (tahmin edilecek değer)

**Veri Kalitesi:**
- ✅ Eksik veri yok (Missing values: 0)
- ✅ Çift kayıtlı veri yok (Duplicates: 0)
- ✅ Veri tam ve temiz

---

## 🔧 Kullanılan Teknolojiler & Kütüphaneler

```python
pandas           # Veri işleme ve analizi
numpy            # Sayısal hesaplamalar
matplotlib       # Veri görselleştirmesi
seaborn          # İstatistiksel grafikler
scikit-learn     # Makine öğrenmesi modelleri
```

**Python Versiyonu:** 3.7+

---

## 📈 Proje Adımları

### 1️⃣ Veri Yükleme ve Keşfetme (EDA)

```python
# Verileri yükleme
df = pd.read_csv("house_price_regression_dataset.csv")

# Veri inceleme
df.head()        # İlk 5 satır
df.shape         # (1000, 8)
df.info()        # Veri türleri ve bilgileri
df.describe()    # İstatistiksel özet
df.isnull().sum() # Eksik değer kontrolü
df.duplicated().sum() # Çift kayıt kontrolü
```

**Çıktı:** Veri set temiz ve 1000x8 boyutlarında

---

### 2️⃣ Korelasyon Analizi

```python
# Heatmap ile korelasyon görselleştirmesi
sns.heatmap(df.corr(), annot=True)
plt.show()

# Yüksek korelasyonlu özellikler tespit etme
def corelletion_drop(df, threshold):
    corr = df.corr()
    liste = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j]) > threshold:
                liste.append(corr.columns[i])
    return liste

# 0.60 eşik değerinin üzerindeki korelasyonlar
high_corr = corelletion_drop(X_train, threshold=0.60)
# Sonuç: [] (Multicollinearity sorunu yok ✅)
```

**Bulgular:**
- Özellikler birbirleriyle düşük korelasyon gösteriyor
- Multicollinearity sorunu yok
- Tüm özellikler modelde kullanılabilir

---

### 3️⃣ Veri Ön İşleme (Preprocessing)

```python
# Hedef değişken ile özellikler ayrılması
X = df.drop("House_Price", axis=1)  # Tahmin değişkenleri
y = df["House_Price"]                # Hedef değişken

# Train-Test Ayrımı (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalizasyon (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**İşlemler:**
- ✅ Train-Test Ayrımı: 80% eğitim, 20% test
- ✅ Normalizasyon: StandardScaler ile ölçekleme
- ✅ Random State: 42 (tekrarlanabilirlik)

---

### 4️⃣ Model Eğitimi ve Değerlendirmesi

Dört farklı regresyon modeli kullanılmıştır:

#### **A) Linear Regression (Doğrusal Regresyon)**

```python
regression = LinearRegression()
lin_model = regression.fit(X_train_scaled, y_train)
y_pred = lin_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)      # 101,434,798.51
print("Mean Squared Error:", mse)       # 101,434,798.51
print("R2 Score:", r2)                  # 0.9984
```

**Sonuç:** 
- MAE: 101,434,798.51
- MSE: 101,434,798.51
- R² Score: **0.9984** ✅

---

#### **B) Ridge Regression (Ridge Düzenleme)**

```python
ridge = Ridge()
lin_model = ridge.fit(X_train_scaled, y_train)
y_pred = lin_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)      # 8,241.59
print("Mean Squared Error:", mse)       # 102,480,990.01
print("R2 Score:", r2)                  # 0.9984
```

**Sonuç:**
- MAE: 8,241.59 ⭐
- MSE: 102,480,990.01
- R² Score: **0.9984** ✅

---

#### **C) Lasso Regression (Lasso Düzenleme)**

```python
lasso = Lasso()
lin_model = lasso.fit(X_train_scaled, y_train)
y_pred = lin_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)      # 8,174.75
print("Mean Squared Error:", mse)       # 101,436,558.18
print("R2 Score:", r2)                  # 0.9984
```

**Sonuç:**
- MAE: 8,174.75 ⭐⭐
- MSE: 101,436,558.18
- R² Score: **0.9984** ✅

---

#### **D) ElasticNet Regression (Elastik Net Düzenleme)**

```python
elastic_net = ElasticNet()
lin_model = elastic_net.fit(X_train_scaled, y_train)
y_pred = lin_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)      # 75,137.54
print("Mean Squared Error:", mse)       # 7,545,488,911.95
print("R2 Score:", r2)                  # 0.8829
```

**Sonuç:**
- MAE: 75,137.54
- MSE: 7,545,488,911.95
- R² Score: **0.8829** ⚠️ (Daha düşük)

---

## 📊 Model Karşılaştırma Tablosu

| Model | MAE | MSE | R² Score | Sonuç |
|-------|-----|-----|----------|-------|
| **Linear Regression** | 101.43M | 101.43M | 0.9984 | İyi |
| **Ridge** | 8,241.59 | 102.48M | 0.9984 | ⭐ Çok İyi |
| **Lasso** | 8,174.75 | 101.43M | 0.9984 | ⭐⭐ En İyi |
| **ElasticNet** | 75,137.54 | 7.55B | 0.8829 | Düşük |

---

## 📝 Metrikler Açıklaması

### **MAE (Mean Absolute Error)**
- Tahmin hatalarının ortalama mutlak değeri
- **Formül:** $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- Daha düşük = Daha iyi
- Birim: Aynı birim (bu durumda $)

### **MSE (Mean Squared Error)**
- Tahmin hatalarının karesi ortalaması
- **Formül:** $MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- Daha düşük = Daha iyi
- Büyük hataları daha ağır penalize eder

### **R² Score (R-Kare)**
- Model açıklayıcılığı (0 ile 1 arasında)
- **Formül:** $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$
- 1.0 = Mükemmel tahmin
- 0.5 = Orta seviye tahmin
- 0.0 = Kötü tahmin

---

## ✨ Sonuçlar ve Öneriler

### 🏆 **En İyi Model: Lasso Regression**

**Neden Lasso?**
- ✅ En düşük MAE: 8,174.75
- ✅ Yüksek R² Score: 0.9984
- ✅ Özellik seçimi yaparak modeli basitleştirir
- ✅ Overfitting riskini azaltır

### 🥈 **İkinci En İyi: Ridge Regression**
- Benzer performans (R² = 0.9984)
- Tüm özellikleri korur
- Daha stabil tahminler

### ⚠️ **ElasticNet Uygun Değil**
- Düşük R² Score: 0.8829
- Yüksek MSE: 7.55 Milyar
- Bu veri seti için uygun değil

---

## 🚀 Projeyi Çalıştırma

### Gereksinimler
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Adımlar
1. `house_price_regression_dataset.csv` dosyasını proje klasörüne yerleştirin
2. `ilk_linear_regression_denemesi.ipynb` dosyasını açın
3. Jupyter Notebook'da sırasıyla hücreleri çalıştırın (Shift+Enter)
4. Grafikler ve sonuçları inceleyin

---

## 📁 Dosya Yapısı

```
├── ilk_linear_regression_denemesi.ipynb
├── house_price_regression_dataset.csv
└── README.md (bu dosya)
```

---

## 💡 Önemli Notlar

- **Veri Kalitesi:** Temiz ve tam veri (outlier/missing value yok)
- **Korelasyon:** Özellikler birbirleriyle düşük korelasyon gösteriyor
- **Tekrarlanabilirlik:** `random_state=42` ile sonuçlar aynı kalır
- **Normalizasyon:** StandardScaler ile tüm özellikler 0-1 aralığında
- **Model Seçimi:** Lasso tercih edilir çünkü daha basit ve etkin

---

## 🔮 Gelecek İyileştirmeler

- [ ] Polinom Regresyon testi
- [ ] Feature Importance analizi
- [ ] Cross-Validation uygulanması (K-Fold)
- [ ] Hiperparametre optimizasyonu (GridSearchCV)
- [ ] Model tuning ve fine-tuning
- [ ] Outlier detection ve removal
- [ ] Feature engineering
- [ ] Model deployment (Web API olarak)

---

## 📚 Kaynaklar

- [Scikit-learn Linear Regression Docs](https://scikit-learn.org/stable/modules/linear_model.html)
- [Ridge, Lasso, ElasticNet](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Tarih:** 2024 | **Dil:** Python 3.7+ | **Versiyon:** 1.0 | **Durum:** ✅ Tamamlandı
