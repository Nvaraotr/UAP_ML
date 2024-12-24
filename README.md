# Prediksi Depresi Menggunakan XGBoost dan Forward Neural Network ✈️ 🏢🏢 💥
Ardhika Yoga Pratama


![Alhamdulillah Sehat](assets/download.jpg)


## Project Overview 🌙
Proyek ini dikembangkan dengan tujuan untuk mendeteksi resiko depresi seseorang. Sistem ini dirancang untuk menganalisis berbagai faktor yang dapat memengaruhi kondisi mental seseorang, dengan ini diharapkan sistem dapat membantu institusi manapun terutama dibidang pendidikan dan mahasiswa itu sendiri dalam mendeteksi tanda-tanda awal masalah depresi sehingga dapat melakukan upaya pencegahan yang lebih awal. Ada 2 hal penting yang diperlukan dalam proyek kali ini yaitu:

- Dataset yang digunakan: [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- Repository proyek: [Repo Sehat Dikit](https://github.com/Nvaraotr/UAP_ML)

## Algoritma Classification ⭐
### Feedforward Neural Network
Feedforward Neural Network (FNN) adalah jenis jaringan saraf tiruan dimana koneksi antara node tidak membentuk siklus. Karakteristik ini membedakannya dari recurrent neural networks (RNN). Jaringan ini terdiri dari input layer, satu atau lebih hidden layers, and an output layer. Informasi mengalir dalam satu arah-dari input ke output-maka dinamakan “feedforward”.
![Arsitektur Model FNN](assets/FNN_Arsitektur.jpg)

### XGBoost
XGBoost (eXtreme Gradient Boosting) adalah pustaka pembelajaran mesin open source terdistribusi yang menggunakan decision trees dengan peningkatan gradien, sebuah algoritma peningkatan pembelajaran terawasi yang memanfaatkan penurunan gradien. Algoritma ini dikenal dengan kecepatan, efisiensi, dan kemampuannya untuk menskalakan dengan baik dengan dataset yang besar.
![Arsitektur XGBoost](assets/Arsitektur_XGBoost.png)

## Preprocessing, EDA, and Modelling 🌍
### Preprocessing
Preprocessing dilakukan untuk menghapus null value, melakukan label encoding dan minmax encoding, serta mengelompokkan isi fitur "Degree" menjadi SMA, Sarjana, Magister, dan Doktor dan ditambahkan pada fitur baru bernama "Education Level"
### EDA
![Distribusi Data](assets/distribusi_data.png)

Dari grafik distribusi data di atas, dapat dilihat bahwa target data memiliki perbedaan yang tidak terlalu jauh sehingga data masih dapat dikatakan balance dan dapat dilanjutkan ke tahap selanjutnya.

### Modelling
#### FNN
```bash
model = Sequential([
    Dense(16, input_dim=x_train.shape[1], activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
![ROC Curves Accuracy](assets/fnn_roc_acc.png)
![ROC Curves Loss](assets/fnn_roc_loss.png)
![FNN Conf Matrix](assets/fnn_conf_matrix.png)
#### XGBoost
```bash
df_xgb = xgb.XGBClassifier()
df_xgb.fit(x_train, y_train)
df_xgb_pred = df_xgb.predict(x_test)
```
![Confussion Matrix](assets/xgb_matrix.png)

Berdasarkan confusion matrix dari FNN dan XGBoost, terlihat bahwa kedua model memiliki performa yang cukup baik, namun terdapat beberapa perbedaan signifikan dalam hasil prediksi mereka. Model XGBoost memiliki True Positive (TP) sebanyak 2.9×10³ dan False Negative (FN) sebanyak 3.9×10², menunjukkan kemampuan yang lebih baik dalam mendeteksi kasus positif dibandingkan FNN yang memiliki TP sebanyak 2.9×10³ tetapi FN lebih tinggi, yaitu 5.1×10². Namun, FNN memiliki True Negative (TN) sebanyak 1.8×10³, setara dengan XGBoost, yang menunjukkan kinerja yang sama dalam mendeteksi kasus negatif.

XGBoost unggul dalam Recall (88.1%) dibandingkan FNN karena lebih sedikit gagal mendeteksi kasus positif (FN lebih rendah). Di sisi lain, FNN memiliki kesalahan lebih tinggi pada False Positive (FP) dibandingkan XGBoost, yang berarti lebih sering salah mendeteksi kasus negatif sebagai positif. Dengan demikian, XGBoost menunjukkan performa yang lebih seimbang antara deteksi positif dan negatif, serta memiliki keunggulan dalam mendeteksi kasus positif, menjadikannya model yang lebih cocok untuk masalah yang membutuhkan sensitivitas tinggi seperti deteksi risiko depresi.

## Streamlit 💻
### Web UI
![Website UI](assets/tampilan_Web.png)
### Result UI
![Result UI](assets/tampilan_Hasil.png)
