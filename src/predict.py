import os
import random
import matplotlib.pyplot as plt
from classifier import CatDogClassifier

ai = CatDogClassifier()
ai.load_trained_model()

if ai.model is None:
    exit()


test_dir = os.path.join(ai.BASE_DIR,"data","test_set")
cats_dir = os.path.join(test_dir,"cats")
dogs_dir = os.path.join(test_dir, "dogs")

random_dir=random.choice([cats_dir,dogs_dir])
image_files = [f for f in os.listdir(random_dir) if f.endswith('.jpg')]
random_image_path = os.path.join(random_dir, random.choice(image_files))


processed_image, original_img = ai.preprocess_image(random_image_path)
prediction_score = ai.model.predict(processed_image)
score_value = prediction_score[0][0]

if score_value > 0.5:
    class_name = "Köpek"
    confidence_score = score_value * 100
else:
    class_name = "Kedi"
    confidence_score = (1 - score_value) * 100


plt.imshow(original_img)
plt.title(f"Tahmin: {class_name}\nTGüven Skoru: %{confidence_score:.2f}")
plt.axis('off')
plt.show()

print(f"Seçilen dosya: {random_image_path}")