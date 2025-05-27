import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 🔧 STEP 1: Set your dataset path here
DATASET_PATH = r"C:\Users\NEXT STORE\Desktop\PlantVillage"

# 📦 STEP 2: Load and preprocess images
def load_data(dataset_path, img_size=(64, 64), max_images_per_class=None):
    images = []
    labels = []
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        count = 0
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)


                images.append(img.flatten())
                labels.append(class_name)
                count += 1
                if max_images_per_class and count >= max_images_per_class:
                    break
            except:
                continue
    return np.array(images), np.array(labels)

# 🔄 STEP 3: Load your dataset
X, y = load_data(DATASET_PATH, img_size=(64, 64))

# 🔠 STEP 4: Encode class labels to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 🎯 STEP 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 🤖 STEP 6: Train KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# 📊 STEP 7: Predict and evaluate
y_pred = knn.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 🔥 Confusion Matrix Plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
