# 🏠 Ev Fiyatı Tahmin Modeli - Doğrusal Regresyon Analizi

## 📋 Proje Açıklaması

Bu Jupyter Notebook projesi, ev fiyatlarını tahmin etmek için **doğrusal regresyon** (Linear Regression) modellerini kullanan bir makine öğrenmesi çalışmasıdır. Proje, farklı regresyon algoritmalarını karşılaştırarak en iyi performansı gösteren modeli belirlemeye yöneliktir.

---

## 📊 Veri Seti

**Dosya:** `house_price_regression_dataset.csv`

**İçeriği:** 1000 ev örneğinin aşağıdaki özelliklerini içerir:
- **Square_Footage:** Evin metrekare cinsinden alanı
- **Num_Bedrooms:** Yatak odası sayısı
- **Num_Bathrooms:** Banyo sayısı
- **Year_Built:** Evin yapım yılı
- **Lot_Size:** Arsa büyüklüğü
- **Garage_Size:** Garaj kapasitesi
- **Neighborhood_Quality:** Mahalle kalitesi puanı
- **House_Price:** 🎯 Tahmin edilecek ev fiyatı (Hedef değişken)

**Veri Kalitesi:** Eksik veri ve çift kayıtlı veri yok ✓

---

## 🔧 Teknolojiler & Kütüphaneler

```
pandas          # Veri işleme
numpy           # Sayısal hesaplamalar
matplotlib      # Görselleştirme
seaborn         # İstatistiksel grafikler
scikit-learn    # Makine öğrenmesi modelleri
```

---

## 📈 Proje Adımları

### 1️⃣ Veri Yükleme ve Keşfetme
- CSV dosyasından veri yükleme
- Veri şekli ve türleri kontrol edildi
- İlk birkaç satırın görüntülenmesi
- İstatistiksel özet analizi

### 2️⃣ Korelasyon Analizi
- Seaborn heatmap ile tüm değişkenlerin korelasyonu incelendi
- Yüksek korelasyona sahip özellikler tespit edildi
- `corelletion_drop()` fonksiyonu ile 0.60 eşik değerinin üzerindeki korelasyonlar kontrol edildi

### 3️⃣ Veri Ön İşleme
- **Train-Test Ayrımı:** 80% eğitim, 20% test (random_state=42)
- **Normalizasyon:** StandardScaler kullanarak özellikler ölçeklendirildi
- **Görselleştirme:** Boxplot ile dağılımlar kontrol edildi

### 4️⃣ Model Eğitimi ve Değerlendirmesi

Dört farklı regresyon modeli kullanılmıştır:

---

## 📊 Model Sonuçları

| Model | MAE | MSE | R² Score |
|-------|-----|-----|----------|
| **Linear Regression** | 101.43M | 101.43M | **0.9984** |
| **Ridge Regression** | 8,241.59 | 102.48M | **0.9984** |
| **Lasso Regression** | 8,174.75 | 101.43M | **0.9984** |
| **ElasticNet** | 75,137.54 | 7.55B | 0.8829 |

---

## 📝 Metrikler Açıklaması

- **MAE (Mean Absolute Error):** Tahmin hatalarının ortalama mutlak değeri
- **MSE (Mean Squared Error):** Tahmin hatalarının karesi ortalaması
- **R² Score:** Model açıklayıcılığı (1.0 = mükemmel)

---

## ✨ Sonuçlar

✅ **En İyi Model:** Lasso Regression
- R² Score: **0.9984** (99.84% açıklayıcılık)
- MAE: **8,174.75** (en düşük hata)
- Özellikleri basitleştirir ve overfitting riskini azaltır

⭐ **İkinci En İyi:** Ridge Regression
- R² Score: **0.9984**
- MAE: **8,241.59**
- Tüm özellikleri korur

⚠️ **ElasticNet:** Daha düşük performans gösterdi (R² = 0.8829)

---

## 🚀 Nasıl Kullanılır?

1. `house_price_regression_dataset.csv` dosyasının aynı klasörde olduğundan emin olun
2. Jupyter Notebook'u açın (`ilk_linear_regression_denemesi.ipynb`)
3. Sırasıyla hücreleri çalıştırın (Shift+Enter)
4. Çıktıları ve grafikleri inceleyin

---

## 💡 Önemli Notlar

- Veri seti temiz (outlier/missing value yok)
- Özellikler birbirleriyle düşük korelasyon gösteriyor (multicollinearity sorunu yok)
- Lasso ve Ridge benzer performans sağlıyor, **Lasso tercih edilir**
- random_state=42 ile sonuçlar tekrarlanabilir

---

## 📁 Dosya Yapısı

```
git_deneme/
├── ilk_linear_regression_denemesi.ipynb
└── README.md
```

---

**Tarih:** 2024 | **Dil:** Python 3.7+ | **Versiyon:** 1.0 | **Durum:** ✅ Tamamlandı