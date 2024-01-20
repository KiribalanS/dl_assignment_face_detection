import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image  # For image processing

# Load images (replace with your actual image paths)
images = []
for i in range(10):
    img = Image.open(f"photo ({i+1}).jpg").resize((150, 150))  # Resize to a consistent size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    images.append(img_array)

# Create labels (1 for your photos, 0 for others)
labels = [1] * 10


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Train the model on our images and labels
history = model.fit(np.array(images), np.array(labels), epochs = 10)



new_img = Image.open("image.jpg").resize((150, 150))
new_img_array = np.array(new_img) / 255.0

prediction = model.predict(np.expand_dims(new_img_array, axis=0))[0][0]

if prediction > 0.5:
    print("This image is likely your photo.")
else:
    print("This image is likely not your photo.")
