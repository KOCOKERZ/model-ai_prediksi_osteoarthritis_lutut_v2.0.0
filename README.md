# Penjelasan kode pada file app.py di atas secara lengkap dan mendetail :

1. **Impor Pustaka yang Diperlukan**:
    ```python
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import os
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from tensorflow.keras.preprocessing import image
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    ```
    - `ImageDataGenerator`: Digunakan untuk melakukan augmentasi gambar seperti rotasi, zoom, dll. Fungsi ini membantu dalam memperbesar dataset untuk meningkatkan kemampuan generalisasi model.
    - `os`: Modul yang menyediakan fungsi untuk berinteraksi dengan sistem operasi seperti memeriksa apakah sebuah path ada.
    - `tensorflow (tf)`: Pustaka utama untuk membangun dan melatih model jaringan saraf.
    - `matplotlib.pyplot (plt)`: Digunakan untuk visualisasi data, seperti plot akurasi dan loss selama pelatihan.
    - `numpy (np)`: Pustaka untuk operasi numerik, digunakan untuk manipulasi array dan matriks.
    - `PIL.Image`: Digunakan untuk membuka dan memanipulasi gambar.
    - `tensorflow.keras.preprocessing.image`: Modul untuk memproses dan memuat gambar.
    - `sklearn.metrics`: Menyediakan fungsi untuk evaluasi performa model seperti matriks kebingungan dan laporan klasifikasi.
    - `seaborn (sns)`: Pustaka untuk visualisasi data yang lebih canggih, digunakan untuk membuat heatmap dari matriks kebingungan.

2. **Mengakses Google Drive**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    - `drive.mount('/content/drive')`: Menghubungkan Google Drive ke lingkungan Colab sehingga file yang ada di Drive bisa diakses dari Colab.

3. **Menentukan Path Dataset**:
    ```python
    data_path = '/content/drive/MyDrive/UAS-AI/Knee-Dataset2/'
    if os.path.exists(data_path):
        print("Path exists")
    else:
        print("Path does not exist")
    ```
    - `data_path`: Variabel yang menyimpan path ke dataset di Google Drive.
    - `os.path.exists(data_path)`: Mengecek apakah path dataset ada atau tidak.

4. **Augmentasi dan Preprocessing Gambar**:
    ```python
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    ```
    - `rescale=1./255`: Menormalkan nilai piksel gambar dari [0, 255] menjadi [0, 1].
    - `rotation_range=20`: Rotasi gambar secara acak dalam rentang 20 derajat.
    - `zoom_range=0.2`: Zoom gambar secara acak hingga 20%.
    - `shear_range=0.2`: Transformasi shear gambar hingga 20%.
    - `fill_mode='nearest'`: Mengisi piksel yang hilang setelah augmentasi dengan nilai piksel terdekat.
    - `validation_split=0.2`: Memisahkan 20% dari data untuk validasi.

5. **Mengatur Directory untuk Data Pelatihan dan Validasi**:
    ```python
    train_dir = data_path

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    ```
    - `train_dir`: Path ke direktori dataset.
    - `target_size=(150, 150)`: Mengubah ukuran gambar menjadi 150x150 piksel.
    - `batch_size=16`: Ukuran batch untuk data pelatihan, yaitu jumlah gambar yang diproses dalam satu iterasi.
    - `batch_size=32`: Ukuran batch untuk data validasi.
    - `class_mode='categorical'`: Mode klasifikasi kategori, yang berarti labelnya berbentuk one-hot encoding.
    - `subset='training'` dan `subset='validation'`: Memisahkan data menjadi set pelatihan dan validasi.

6. **Membangun Model Jaringan Saraf Konvolusional**:
    ```python
    num_classes = len(train_generator.class_indices)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    ```
    - `num_classes`: Jumlah kelas yang akan diklasifikasikan.
    - `tf.keras.models.Sequential`: Model sequential, yang berarti layer-layer ditambahkan secara berurutan.
    - `tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3))`: Layer konvolusi dengan 64 filter, ukuran kernel 3x3, dan fungsi aktivasi ReLU. `input_shape` mendefinisikan bentuk input yaitu gambar berukuran 150x150 dengan 3 kanal warna (RGB).
    - `tf.keras.layers.MaxPooling2D(2, 2)`: Layer pooling untuk mengurangi ukuran dimensi gambar sebanyak 2x2.
    - `tf.keras.layers.Dropout(0.5)`: Layer dropout untuk mengurangi overfitting dengan mengabaikan 50% unit selama pelatihan.
    - `tf.keras.layers.Flatten()`: Mengubah data dari bentuk 2D ke 1D.
    - `tf.keras.layers.Dense(512, activation='relu')`: Layer fully connected dengan 512 unit dan fungsi aktivasi ReLU.
    - `tf.keras.layers.Dense(num_classes, activation='softmax')`: Layer output dengan jumlah unit sebanyak jumlah kelas dan fungsi aktivasi softmax untuk mengeluarkan probabilitas kelas.

7. **Callback untuk Menghentikan Pelatihan Ketika Akurasi Mencapai 90%**:
    ```python
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') > 0.90):
                print("\nAkurasi telah mencapai >90%!")
                self.model.stop_training = True

    callbacks = myCallback()
    ```
    - `tf.keras.callbacks.Callback`: Kelas dasar untuk membuat callback custom.
    - `on_epoch_end(self, epoch, logs={})`: Fungsi yang dipanggil pada akhir setiap epoch. Jika akurasi lebih dari 90%, pelatihan dihentikan.

8. **Kompilasi Model**:
    ```python
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    ```
    - `optimizer=tf.optimizers.Adam(learning_rate=0.0001)`: Menggunakan optimizer Adam dengan learning rate 0.0001.
    - `loss='categorical_crossentropy'`: Menggunakan loss function categorical crossentropy untuk klasifikasi multi-kelas.
    - `metrics=['accuracy']`: Menggunakan metrik akurasi untuk evaluasi model.

9. **Melatih Model**:
    ```python
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        verbose=2,
        callbacks=[callbacks]
    )
    ```
    - `model.fit`: Melatih model dengan data pelatihan dan validasi.
    - `epochs=50`: Jumlah epoch pelatihan, yaitu 50 kali iterasi penuh melalui dataset.
    - `verbose=2`: Menampilkan log pelatihan dengan format yang lebih ringkas.
    - `callbacks=[callbacks]`: Menggunakan callback custom yang telah didefinisikan sebelumnya.

10. **Menyimpan Model**:
    ```python
    model.save('model.h5')
    ```
    - `model.save('model.h5')`: Menyimpan model terlatih dalam format HDF5.

11. **Visualisasi Akurasi dan Loss**:
    ```python
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Akurasi Model')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt

.title('Loss Model')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    ```
    - `plt.plot`: Membuat plot untuk akurasi dan loss selama pelatihan.
    - `plt.legend`: Menambahkan legenda untuk plot.

12. **Menampilkan Akurasi dan Loss**:
    ```python
    print("Training Accuracy:", history.history['accuracy'])
    print("Validation Accuracy:", history.history['val_accuracy'])
    print("Training Loss:", history.history['loss'])
    print("Validation Loss:", history.history['val_loss'])
    ```

13. **Label Indeks Kelas**:
    ```python
    label_to_index = train_generator.class_indices
    print(label_to_index)
    ```
    - `train_generator.class_indices`: Mendapatkan indeks kelas untuk setiap label kelas.

14. **Memuat Model yang Disimpan**:
    ```python
    model = tf.keras.models.load_model('model.h5')
    ```
    - `tf.keras.models.load_model('model.h5')`: Memuat model yang telah disimpan sebelumnya.

15. **Fungsi untuk Memproses Gambar**:
    ```python
    def process_image(img_path):
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    ```
    - `image.load_img(img_path, target_size=(150, 150))`: Memuat gambar dari path yang diberikan dan mengubah ukurannya menjadi 150x150 piksel.
    - `image.img_to_array(img)`: Mengubah gambar yang dimuat menjadi array numpy.
    - `np.expand_dims(img_array, axis=0)`: Menambahkan dimensi baru pada array untuk mencocokkan bentuk input model (batch size).
    - `img_array /= 255.0`: Menormalisasi nilai piksel gambar dari [0, 255] menjadi [0, 1].

16. **Fungsi untuk Memprediksi Gambar**:
    ```python
    def predict_image(img_path):
        img_array = process_image(img_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        return predicted_class, predictions
    ```
    - `img_array = process_image(img_path)`: Memproses gambar menggunakan fungsi `process_image`.
    - `predictions = model.predict(img_array)`: Meminta model untuk memprediksi kelas gambar yang sudah diproses.
    - `predicted_class = np.argmax(predictions, axis=1)`: Mendapatkan kelas yang diprediksi dengan probabilitas tertinggi.

17. **Fungsi untuk Menampilkan Gambar dengan Prediksi**:
    ```python
    def show_image_with_prediction(img_path):
        img = Image.open(img_path)
        predicted_class, predictions = predict_image(img_path)

        label_to_index = train_generator.class_indices
        index_to_label = {v: k for k, v in label_to_index.items()}
        predicted_label = index_to_label[predicted_class[0]]

        plt.imshow(img)
        plt.title(f'Predicted class: {predicted_label}')
        plt.axis('off')
        plt.show()

        print(f"Predicted class: {predicted_label}")
        print(f"Predictions: {predictions}")
    ```
    - `img = Image.open(img_path)`: Membuka gambar dari path yang diberikan.
    - `predicted_class, predictions = predict_image(img_path)`: Mendapatkan prediksi kelas dan probabilitas dari gambar yang diproses.
    - `label_to_index = train_generator.class_indices`: Mendapatkan indeks kelas untuk setiap label kelas.
    - `index_to_label = {v: k for k, v in label_to_index.items()}`: Membuat kamus untuk konversi dari indeks ke label kelas.
    - `plt.imshow(img)`: Menampilkan gambar menggunakan Matplotlib.
    - `plt.title(f'Predicted class: {predicted_label}')`: Menambahkan judul pada gambar yang berisi kelas yang diprediksi.
    - `plt.axis('off')`: Menghilangkan sumbu pada plot gambar.
    - `plt.show()`: Menampilkan plot gambar.
    - `print(f"Predicted class: {predicted_label}")`: Mencetak kelas yang diprediksi.
    - `print(f"Predictions: {predictions}")`: Mencetak probabilitas prediksi untuk setiap kelas.

18. **Menampilkan Gambar dengan Prediksi**:
    ```python
    img_path = '/content/drive/MyDrive/KNEE_X-RAY/download.jpeg'
    show_image_with_prediction(img_path)
    ```
    - `img_path`: Path ke gambar yang akan diprediksi.
    - `show_image_with_prediction(img_path)`: Memanggil fungsi untuk menampilkan gambar dan prediksinya.

19. **Fungsi untuk Membuat dan Menampilkan Matriks Kebingungan**:
    ```python
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    ```
    - `cm`: Matriks kebingungan.
    - `classes`: Daftar label kelas.
    - `normalize=False`: Parameter untuk menentukan apakah matriks kebingungan akan dinormalisasi atau tidak.
    - `title='Confusion Matrix'`: Judul plot matriks kebingungan.
    - `cmap=plt.cm.Blues`: Skema warna yang digunakan untuk heatmap.
    - `cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]`: Normalisasi matriks kebingungan dengan membagi setiap nilai dengan jumlah nilai pada barisnya.
    - `plt.figure(figsize=(10, 7))`: Mengatur ukuran plot.
    - `sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=classes, yticklabels=classes)`: Membuat heatmap dari matriks kebingungan menggunakan Seaborn.

20. **Menghitung Matriks Kebingungan**:
    ```python
    validation_steps = validation_generator.samples // validation_generator.batch_size
    y_true = validation_generator.classes
    y_pred = model.predict(validation_generator, steps=validation_steps)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    class_labels = list(label_to_index.keys())

    plot_confusion_matrix(cm, class_labels)
    ```
    - `validation_steps`: Jumlah langkah validasi, dihitung sebagai jumlah sampel validasi dibagi dengan ukuran batch validasi.
    - `y_true`: Label sebenarnya dari data validasi.
    - `y_pred`: Prediksi model untuk data validasi.
    - `y_pred_classes`: Kelas yang diprediksi oleh model, diambil dari probabilitas prediksi.
    - `cm = confusion_matrix(y_true, y_pred_classes)`: Membuat matriks kebingungan dari label sebenarnya dan label prediksi.
    - `class_labels = list(label_to_index.keys())`: Daftar label kelas.

21. **Menampilkan Laporan Klasifikasi**:
    ```python
    print("Classification Report")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))
    ```
    - `classification_report(y_true, y_pred_classes, target_names=class_labels)`: Membuat laporan klasifikasi yang mencakup metrik seperti precision, recall, dan F1-score untuk setiap kelas.