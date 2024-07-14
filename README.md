# Penjelasan 

Kode di atas merupakan implementasi lengkap dari sebuah program yang digunakan untuk melatih model deep learning menggunakan dataset gambar, melakukan prediksi, dan menampilkan hasil prediksi pada gambar. Kode ini menggunakan pustaka TensorFlow dan Keras untuk melatih model convolutional neural network (CNN) pada dataset gambar lutut. Berikut penjelasan rinci setiap bagian kode:

### Import Library
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
```
- `tensorflow.keras.preprocessing.image.ImageDataGenerator`: Digunakan untuk augmentasi gambar secara real-time.
- `os`: Modul untuk berinteraksi dengan sistem operasi, misalnya untuk memeriksa keberadaan folder.
- `tensorflow as tf`: Library utama untuk deep learning.
- `matplotlib.pyplot as plt`: Digunakan untuk plotting grafik.
- `numpy as np`: Digunakan untuk operasi numerik.
- `PIL.Image`: Digunakan untuk memanipulasi gambar.
- `tensorflow.keras.preprocessing.image`: Digunakan untuk memproses gambar.

### Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
- `drive.mount('/content/drive')`: Mount Google Drive agar bisa diakses dari Google Colab.

### Path Dataset
```python
data_path = '/content/drive/MyDrive/UAS-AI/Knee-Dataset2/'
if os.path.exists(data_path):
    print("Path exists")
else:
    print("Path does not exist")
```
- `data_path`: Path menuju folder dataset gambar di Google Drive.
- `os.path.exists(data_path)`: Mengecek apakah path dataset ada atau tidak.

### Augmentasi Data
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
- `rescale=1./255`: Mengubah skala pixel dari [0, 255] menjadi [0, 1].
- `rotation_range=20`: Memutar gambar secara acak dalam range 20 derajat.
- `zoom_range=0.2`: Zoom gambar secara acak hingga 20%.
- `shear_range=0.2`: Shear gambar secara acak hingga 20%.
- `fill_mode='nearest'`: Mengisi pixel yang hilang setelah augmentasi dengan nilai pixel terdekat.
- `validation_split=0.2`: Membagi dataset menjadi 80% training dan 20% validation.

### Directory Dataset
```python
train_dir = data_path
```
- `train_dir`: Menyimpan path directory dataset.

### Generator Data Training dan Validation
```python
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
- `flow_from_directory`: Mengambil gambar dari folder dan menerapkan augmentasi.
- `target_size=(150, 150)`: Mengubah ukuran gambar menjadi 150x150 pixel.
- `batch_size=16` dan `batch_size=32`: Jumlah gambar per batch.
- `class_mode='categorical'`: Mode klasifikasi kategori.
- `subset='training'` dan `subset='validation'`: Menentukan subset untuk training dan validation.

### Jumlah Kelas
```python
num_classes = len(train_generator.class_indices)
```
- `num_classes`: Menyimpan jumlah kelas dalam dataset.

### Arsitektur Model
```python
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
- `Sequential`: Model berlapis linear.
- `Conv2D`: Layer konvolusi dengan 64 filter ukuran 3x3 dan fungsi aktivasi ReLU.
- `MaxPooling2D`: Layer pooling dengan ukuran pool 2x2.
- `Dropout`: Regularisasi dropout dengan rate 0.5.
- `Flatten`: Mengubah output dari layer konvolusi menjadi 1D.
- `Dense`: Layer fully connected dengan 512 unit dan fungsi aktivasi ReLU.
- `Dense(num_classes, activation='softmax')`: Layer output dengan jumlah unit sesuai jumlah kelas dan fungsi aktivasi softmax.

### Callback untuk Menghentikan Training
```python
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.90):
            print("\nAkurasi telah mencapai >90%!")
            self.model.stop_training = True

callbacks = myCallback()
```
- `myCallback`: Callback custom untuk menghentikan training jika akurasi > 90%.

### Kompilasi Model
```python
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```
- `optimizer=tf.optimizers.Adam(learning_rate=0.0001)`: Optimizer Adam dengan learning rate 0.0001.
- `loss='categorical_crossentropy'`: Fungsi loss untuk klasifikasi kategori.
- `metrics=['accuracy']`: Metrik akurasi untuk evaluasi.

### Training Model
```python
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    verbose=2,
    callbacks=[callbacks]
)
```
- `model.fit`: Melatih model.
- `epochs=50`: Jumlah epoch training.
- `verbose=2`: Menampilkan output training.
- `callbacks=[callbacks]`: Menggunakan callback custom.

### Menyimpan Model
```python
model.save('model.h5')
```
- `model.save('model.h5')`: Menyimpan model ke file 'model.h5'.

### Plot Akurasi dan Loss
```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Akurasi Model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print("Training Accuracy:", history.history['accuracy'])
print("Validation Accuracy:", history.history['val_accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print("Training Loss:", history.history['loss'])
print("Validation Loss:", history.history['val_loss'])
```
- `plt.plot`: Plotting grafik akurasi dan loss.
- `plt.show()`: Menampilkan grafik.
- `history.history['accuracy']` dan `history.history['val_accuracy']`: Akurasi training dan validation.
- `history.history['loss']` dan `history.history['val_loss']`: Loss training dan validation.

### Label ke Indeks
```python
label_to_index = train_generator.class_indices
print(label_to_index)
```
- `train_generator.class_indices`: Menampilkan mapping label ke indeks.

### Memuat Model yang Disimpan
```python
model = tf.keras.models.load_model('model.h5')
```
- `tf.keras.models.load_model('model.h5')`: Memuat model yang disimpan.

### Proses dan Prediksi Gambar
```python
def process_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(img_path):
    img_array = process_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class, predictions
```
- `process_image`: Memproses gambar dari path menjadi array yang siap diprediksi.
- `predict_image`: Melakukan prediksi pada gambar.

### Menampilkan Gambar dengan Prediksi
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

img_path = '/content/drive/MyDrive/KNEE_X-RAY/download.jpeg'
show_image_with_prediction(img_path)
```
- `show_image_with_prediction`: Menampilkan gambar dengan label prediksi.