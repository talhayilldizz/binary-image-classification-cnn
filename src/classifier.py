import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

class CatDogClassifier:
    def __init__(self):
        self.IMG_SIZE = 128
        self.BATCH_SIZE = 32
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_PATH = os.path.join(self.BASE_DIR, 'models', 'kedi_kopek_modeli.keras')
        self.model = None

    def build_model(self):
        self.model = Sequential()

        self.model.add(
            Conv2D(
                32, (3,3),
                activation='relu',
                input_shape = (128,128,3)
            )
        )
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(
            Conv2D(
                64, (3,3),
                activation='relu'
            )
        )
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(
            Conv2D(
                128,(3,3),
                activation='relu'
            )
        )
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1,activation='sigmoid'))

        self.model.compile(
            optimizer = "adam",
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )


    def train(self, epochs=5):
        train_dir = os.path.join(self.BASE_DIR, "data", "training_set")
        test_dir = os.path.join(self.BASE_DIR, "data", "test_set")

        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range = 0.2,
            horizontal_flip = True
        )
        
        train_data = datagen.flow_from_directory(
            train_dir, target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE, class_mode='binary'
        )

        val_data = datagen.flow_from_directory(
            test_dir, target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE, class_mode="binary"
        )

        self.build_model()

        history = self.model.fit(
            train_data, epochs=epochs, validation_data=val_data
        )

        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        self.model.save(self.MODEL_PATH)
        print(f"Model başarıyla kaydedildi: {self.MODEL_PATH}")
        self.plot_training_history(history)

    def load_trained_model(self):
        if os.path.exists(self.MODEL_PATH):
            self.model = tf.keras.models.load_model(self.MODEL_PATH)
        else:
            print("Hata: Kayıtlı model bulunamadı!")

    def preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(self.IMG_SIZE, self.IMG_SIZE))
        x = image.img_to_array(img)
        x /= 255.0
        return np.expand_dims(x, axis=0), img
    
    def plot_training_history(self, history):
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        epochs_range = range(1,len(acc) + 1)

        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.plot(epochs_range, acc, label='Eğitim Başarısı (Training Acc)')
        plt.plot(epochs_range, val_acc, label='Test Başarısı (Validation Acc)')
        plt.title('Eğitim ve Test Başarısı')
        plt.xlabel('Epoch')
        plt.ylabel('Doğruluk (Accuracy)')
        plt.legend(loc='lower right')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Eğitim Hatası (Training Loss)')
        plt.plot(epochs_range, val_loss, label='Test Hatası (Validation Loss)')
        plt.title('Eğitim ve Test Hatası')
        plt.xlabel('Epoch')
        plt.ylabel('Hata (Loss)')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
