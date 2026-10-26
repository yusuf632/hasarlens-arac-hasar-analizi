# HasarLens — Araç Hasar Analizi

HasarLens, bir araç fotoğrafını sürükleyip bırakarak veya seçerek analiz etmenizi sağlayan yerel web uygulamasıdır. Fotoğraf, projedeki MobileNetV2 modeline gönderilir ve dört sınıftan biri için tahmin ile güven oranı gösterilir:

- Arka hasarlı
- Arka hasarsız
- Ön hasarlı
- Ön hasarsız

> Bu model genel amaçlı nesne tanıma yapmaz. Yalnızca eğitim verisindeki araç fotoğraflarına benzer görüntüler için ön/arka ve hasarlı/hasarsız sınıflandırması üretir. Sonuç, fotoğrafa dayalı bir ön değerlendirmedir; ekspertiz veya profesyonel inceleme yerine geçmez.

## Özellikler

- Fotoğraf seçme, sürükleyip bırakma ve önizleme
- Olası hasar durumu, tahmin edilen araç bölgesi ve dört sınıf için olasılık dağılımı
- JPEG, PNG ve WEBP desteği; 10 MB yükleme sınırı
- Görseller analiz için bellekte işlenir ve sunucuya kaydedilmez
- Model yerel dosyadan yüklenir; analiz sırasında harici yapay zekâ servisine istek gönderilmez
- Mevcut grafik raporlu komut satırı aracı korunmuştur

## Kurulum ve çalıştırma

Python 3.10 veya üstü bir sanal ortam önerilir. TensorFlow'un kullandığınız Python sürümüyle uyumlu olduğundan emin olun.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Ardından tarayıcıda [http://127.0.0.1:5000](http://127.0.0.1:5000) adresini açın. Sunucu ilk model kullanımında `model_arac_hasar_son.keras` modelini belleğe yükler. Model dosyası uygulama dosyalarıyla aynı klasörde olmalıdır.

Uygulamayı durdurmak için PowerShell penceresinde `Ctrl+C` tuşlarına basın.

## API

`POST /api/analyze` endpoint'ine `image` adlı multipart dosya alanıyla JPEG, PNG veya WEBP görseli gönderin. Başarılı yanıt sınıfı, tahmin edilen bölgeyi, güven yüzdesini, dört sınıfın olasılıklarını ve analiz zamanını JSON olarak döndürür. Geçersiz görsel ve model yükleme hataları sırasıyla `400` ve `503` durum kodlarıyla bildirilir.

## Modeli yeniden eğitme

Eğitim verileri bu depoda boyut nedeniyle yayımlanmamıştır. Yeniden eğitim için görselleri yerel olarak `aracVeriSeti/Dataset` altındaki `ArkaHasarlı`, `ArkaHasarsız`, `OnHasarlı`, `OnHasarsız` klasörlerine yerleştirin. Her görsel yalnızca doğru sınıf klasöründe olmalıdır. Eğitim, doğrulama ve bağımsız test kümeleri sınıf başına ayrı ayrı oluşturulur; birebir kopya görseller aynı sınıfta tekilleştirilir. Farklı sınıflarda aynı görsel bulunursa eğitim durdurulur.

```powershell
python train_model.py
```

Eğitim mevcut modelden devam eder; hafif görüntü artırma, sınıf ağırlıkları, erken durdurma ve MobileNetV2'nin son katmanlarında ince ayar uygular. Yeni model bağımsız test kümesinde hem makro F1 hem doğruluk ölçütlerinde mevcut modeli geçerse etkin model güncellenir; aksi durumda mevcut model korunur. Eğitim yerel TensorFlow donanım desteğine bağlıdır ve GPU olmayan sistemlerde CPU üzerinde daha uzun sürebilir.

Mevcut model, bu veri kümesinden ayrılan 335 görsellik test bölümünde %90,45 doğruluk ve %88,05 makro F1 elde etti. Bu sonuç yalnızca aynı veri kaynağından ayrılan test fotoğrafları için geçerlidir; farklı kamera, açı veya hasar türlerinde aynı başarıyı garanti etmez.

Model yalnızca bu dört sınıftan birini tahmin eder; hasarın yerini fotoğraf üzerinde işaretlemez, şiddetini ölçmez ve eğitim verisinde temsil edilmeyen fotoğraflar için güvenilir sonuç garantisi vermez.

## Komut satırı raporu

Grafik rapor aracını çalıştırmak için:

```powershell
python arabaHasarTespiti.py
```

Varsayılan örnek görsel `Test13.jpg` dosyasıdır.

## Teknolojiler

Python, Flask, TensorFlow/Keras, MobileNetV2, Pillow, NumPy ve Matplotlib.
