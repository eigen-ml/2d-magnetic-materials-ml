# Demo nasıl çalışır

Tek dosyalık bir Streamlit uygulaması (`demo/app.py`). Tez modelini ve revize modeli yan yana gösterir. Modeli yeniden eğitmez. Ekrandaki bütün sayılar repodaki dosyalardan okunur.

## Çalıştırma

İlk kurulum (bir kez, internet gerekir):

```
python -m venv .venv
.venv\Scripts\python -m pip install -r demo/requirements.txt
```

Çalıştırma (repo kökünden, internet gerekmez):

```
.venv\Scripts\streamlit run demo/app.py
```

Tarayıcı `http://localhost:8501` adresini açar. İlk açılışta V2DB etiketleri hazırlanırken yaklaşık 20 saniye beklenir, sonra önbellekten gelir.

## Kütüphaneler ve görevleri

| Kütüphane | Ne için |
|---|---|
| streamlit | Python kodundan web arayüzü: sekmeler, tablo, metin kutusu |
| scikit-learn | Modeller `GradientBoostingClassifier`. Sürüm 1.7.2 olmalı, modeller bu sürümle kaydedildi |
| joblib | `model.joblib` dosyalarını yükler (içeride pickle kullanır, yalnız kendi dosyalarımızı yüklüyoruz) |
| pymatgen | Formülü okur, indirgenmiş formüle çevirir, element miktarlarını verir |
| pandas, numpy | Tablolar ve skor hesabı |
| matplotlib | Adaylar sekmesindeki histogram |

## Parçalar

**Modelleri yükleme.** `scripts/04_discover.py` içindeki `load_model` iki klasör için çağrılır: `models/legacy_hybrid/` (tez) ve `models/` (revize). Her biri model, eşik (`threshold.npy`) ve `metadata.json` döndürür. `@st.cache_resource` sayesinde bu iş bir kez yapılır.

**Formülü özellik vektörüne çevirme.** İki model aynı fikri kullanır, ama özellik listeleri farklıdır:

- Tez modeli: 64 özellik (63 element oranı + `natoms`). Kod, `thesis-2026` etiketindeki `scripts/04_discover.py` dosyasından kopyalandı (`featurize_thesis`).
- Revize model: 52 özellik. Kod, `api/main.py` ile aynı şekilde `scripts/04_discover.py` dosyasından gelir (`discover.featurize`).

Örnek: `MnFeBr3` için pymatgen `{Mn: 1, Fe: 1, Br: 3}` verir. Toplam 5 atom. Özellikler `elem_Mn = 0.2`, `elem_Fe = 0.2`, `elem_Br = 0.6`, `natoms = 5`, diğer bütün elementler 0. Sütun sırası `metadata.json` içindeki `feature_names` listesinden alınır, çünkü model eğitimdeki sırayı bekler.

**Bilinmeyen element kontrolü.** Tez sürümündeki kod, modelin hiç görmediği bir elementi sessizce atıp yine skor üretiyordu. Demo önce `unknown_elements` ile bakar. Böyle bir element varsa o model için skor üretmez ve uyarı gösterir.

**Veri setindeki etiket.** Tez modeli için `data/legacy_hybrid/combined_dataset.csv` (etiket ve kaynak: C2DB veya V2DB). Revize model için `data/v2db.csv`. Aynı formül V2DB'de hem manyetik hem manyetik değil olarak geçiyorsa revize eğitimde kullanılmamıştı (`scripts/02_prepare_data.py`). Demo bunu "çelişkili etiket" diye yazar.

**Adaylar sekmesi.** `results/legacy/novel_candidates.csv` içinden `confidence` sütunu "Very High" olanlar (tez scriptinde skor 0,95 ve üstü). Sayı dosyadan hesaplanır ve `discovery_summary.json` ile yan yana gösterilir. "Yeniden hesapla" düğmesi 1.472 formülü tez modeliyle yeniden skorlar ve dosyadaki skorla en büyük farkı yazar.

**Model kartı.** Metrikler ve karışıklık matrisi `metadata.json` dosyalarından, önemli özellikler `feature_importance.csv` dosyalarından, leave-element-out tablosu `results/grouped_validation.json` dosyasından okunur.

## Ayarlar

`.streamlit/config.toml`: kullanım istatistiği gönderimi kapalı, açık tema, uygulama yalnız `localhost` üzerinden açılır.
