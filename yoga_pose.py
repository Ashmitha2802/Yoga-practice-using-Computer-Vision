import cv2
import mediapipe as mp
import math
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===================== PATHS =====================
train_dir = r'C:\Users\anant\Desktop\Mini_Pro_One\content\dataset\train'
val_dir   = r'C:\Users\anant\Desktop\Mini_Pro_One\content\dataset\validation'

# ===================== PARAMETERS =====================
image_size = (150, 150)
batch_size = 16
epochs = 50

# ===================== DATA GENERATORS =====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ===================== CNN MODEL =====================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===================== CALLBACKS =====================
callbacks = [
    ModelCheckpoint('yoga_pose_model.h5', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# ===================== TRAIN MODEL =====================
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks
)

# ===================== EVALUATION =====================
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# ===================== ACCURACY GRAPH =====================
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# ===================== CONFUSION MATRIX =====================
print("Generating Confusion Matrix...")

y_true = validation_generator.classes
start_time = time.time()

y_pred_probs = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

print(f"Prediction Time: {time.time() - start_time:.2f} seconds")

print(classification_report(
    y_true,
    y_pred,
    target_names=list(validation_generator.class_indices.keys())
))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=validation_generator.class_indices.keys(),
    yticklabels=validation_generator.class_indices.keys()
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ===================== POSE FEEDBACK USING MEDIAPIPE =====================
def process_pose(image_path, ideal_landmarks):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        detected = {
            "left_knee": (
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
            ),
            "right_knee": (
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
            ),
            "left_elbow": (
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
            ),
            "right_shoulder": (
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            )
        }

        feedback = {}
        for part, (dx, dy) in detected.items():
            ix, iy = ideal_landmarks[part]
            dist = math.sqrt((dx - ix) ** 2 + (dy - iy) ** 2)
            accuracy = max(0, 100 - (dist * 500))
            feedback[part] = round(accuracy, 2)

        return feedback

    return "No pose detected"

# ===================== IDEAL TREE POSE =====================
ideal_tree_pose = {
    "left_knee": (0.3, 0.6),
    "right_knee": (0.7, 0.6),
    "left_elbow": (0.4, 0.3),
    "right_shoulder": (0.8, 0.3)
}

image_path = r"C:\Users\admin\OneDrive\Desktop\final_mini_project\content\dataset\validation\WarriorPose\00000001.jpg"

feedback = process_pose(image_path, ideal_tree_pose)
print("Pose Feedback:", feedback)
