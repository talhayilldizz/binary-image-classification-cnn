# CNN ile Kedi ve Köpek Görüntü Sınıflandırma

Bu proje, **Derin Öğrenme** ve **Evrişimli Sinir Ağları** kullanılarak kedi ve köpek görüntülerini ikili olarak sınıflandırmak amacıyla geliştirilmiştir.

Proje başlangıçta Kaggle ortamında geliştirilmiş, daha sonra **Nesne Yönelimli Programlama** prensiplerine uygun şekilde modüler bir yapıya dönüştürülerek yerel geliştirme ortamına taşınmıştır.

## Kullanılan Teknolojiler

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn

## Kullanılan Veri Seti

Bu projede kullanılan veri seti Kaggle üzerindeki **Dogs vs Cats** veri setidir.

Dataset Linki:  
https://www.kaggle.com/datasets/tongpython/cat-and-dog

> Dataset repository içinde yer almamaktadır.  
> Lütfen indirip `data/` klasörü içine aşağıdaki yapıya uygun şekilde yerleştiriniz:


## Eğitim Sonuçları
<img width="1200" height="500" alt="model_performance3" src="https://github.com/user-attachments/assets/1a7d8d84-aa02-47af-948c-b9d53968db88" />

### Eğitim ve Test Doğruluğu

Model eğitim sürecinde doğruluk değerleri düzenli artış göstermiştir. Validation loss’un son epochta hafif yükselmesi modelin overfitting eğilimine girmeye başladığını göstermektedir. Bu nedenle EarlyStopping ve veri artırma teknikleri ile performans iyileştirmesi planlanabilir.






<img width="640" height="480" alt="predict_1" src="https://github.com/user-attachments/assets/b8dd2027-e859-489c-997d-640b19ce919b" />



<img width="640" height="480" alt="predict_2" src="https://github.com/user-attachments/assets/71d8db2b-3481-414f-905c-45eeccebae34" />
