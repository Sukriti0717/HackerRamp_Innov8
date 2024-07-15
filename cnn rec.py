#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("Starting the recommendation system process...")

# Set random seed for reproducibility
np.random.seed(42)
print("Random seed set for reproducibility.")

# Paths and constants
RATINGS_PATH = r"D:\\myntradata\\Images\\imgrating5.csv"
IMAGES_FOLDER = r"D:\\myntradata\\Images\\Images"
IMAGE_SIZE = (224, 224)
EMBEDDING_DIM = 32
BATCH_SIZE = 64
EPOCHS = 50

print(f"Constants set: Image size {IMAGE_SIZE}, Embedding dimension {EMBEDDING_DIM}, Batch size {BATCH_SIZE}, Epochs {EPOCHS}")

# Load ratings data
print("Loading ratings data...")
ratings_df = pd.read_csv(RATINGS_PATH)
print("Ratings data loaded successfully.")

# Create user and item mappings
print("Creating user and item mappings...")
user_ids = ratings_df['user_id'].unique()
item_ids = ratings_df['image_name'].unique()
user_to_index = {id: index for index, id in enumerate(user_ids)}
item_to_index = {id: index for index, id in enumerate(item_ids)}
print(f"Mappings created. Number of users: {len(user_ids)}, Number of items: {len(item_ids)}")

# Convert user_ids and item_ids to indices
ratings_df['user_index'] = ratings_df['user_id'].map(user_to_index)
ratings_df['item_index'] = ratings_df['image_name'].map(item_to_index)
print("User and item IDs converted to indices.")

# Preprocess ratings
print("Preprocessing ratings...")
scaler = StandardScaler()
ratings_df['rating_scaled'] = scaler.fit_transform(ratings_df[['rating']])
print("Ratings preprocessed and scaled.")

# Load and preprocess images
def load_and_preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=IMAGE_SIZE)
        return img_to_array(img) / 255.0
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

# Create the pre-trained CNN model
print("Creating pre-trained CNN model...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + (3,))
for layer in base_model.layers:
    layer.trainable = False
image_features = Flatten()(base_model.output)
cnn_model = Model(inputs=base_model.input, outputs=image_features)
print("Pre-trained CNN model created successfully.")


#import os
import re

# ... (previous code remains the same)

# Extract features for all images
print("Extracting features for all images...")
image_features = {}
for item_id in item_ids:
    # Use regex to find any file starting with the item_id
    pattern = re.compile(f"^{re.escape(item_id)}.*")
    matching_files = [f for f in os.listdir(IMAGES_FOLDER) if pattern.match(f)]
    
    if matching_files:
        image_path = os.path.join(IMAGES_FOLDER, matching_files[0])
        print(f"Processing image: {image_path}")
        image = load_and_preprocess_image(image_path)
        if image is not None:
            image_features[item_id] = cnn_model.predict(np.expand_dims(image, axis=0)).flatten()
    else:
        print(f"No matching image file found for item_id: {item_id}")

print(f"Features extracted for {len(image_features)} images.")

# Check if any images were processed
if len(image_features) == 0:
    raise ValueError("No images were successfully processed. Please check your image files and paths.")

# ... (rest of the code remains the same)


# Remove ratings for items without images
ratings_df = ratings_df[ratings_df['image_name'].isin(image_features.keys())]
print(f"Ratings dataframe updated. New shape: {ratings_df.shape}")

# Prepare data for model
print("Preparing data for the model...")
X_user = ratings_df['user_index'].values
X_item = ratings_df['item_index'].values
X_img = np.array([image_features[item_id] for item_id in ratings_df['image_name']])
y = ratings_df['rating_scaled'].values
print("Data preparation completed.")

n_samples = len(X_user)
print(f"Total number of samples: {n_samples}")

# Split the data into train, validation, and test sets
print("Splitting data into train, validation, and test sets...")
X_user_train_val, X_user_test, X_item_train_val, X_item_test, X_img_train_val, X_img_test, y_train_val, y_test = train_test_split(
    X_user, X_item, X_img, y,n_samples, test_size=0.2,train_size=0.8, random_state=42)

X_user_train, X_user_val, X_item_train, X_item_val, X_img_train, X_img_val, y_train, y_val = train_test_split(
    X_user_train_val, X_item_train_val, X_img_train_val, y_train_val,n_samples, test_size=0.25,train_size=0.8, random_state=42)  # 0.25 x 0.8 = 0.2
print("Data split completed.")

# Create the recommendation model
def create_recommendation_model(num_users, num_items, img_features_dim):
    print("Creating recommendation model...")
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    img_input = Input(shape=(img_features_dim,))
    
    user_embedding = Embedding(num_users, EMBEDDING_DIM)(user_input)
    item_embedding = Embedding(num_items, EMBEDDING_DIM)(item_input)
    
    x = Concatenate()([Flatten()(user_embedding), Flatten()(item_embedding), img_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)
    
    model = Model(inputs=[user_input, item_input, img_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    print("Recommendation model created successfully.")
    return model

# Create and train the model
model = create_recommendation_model(len(user_ids), len(item_ids), X_img.shape[1])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
print("Training the model...")
history = model.fit(
    [X_user_train, X_item_train, X_img_train], y_train,
    validation_data=([X_user_val, X_item_val, X_img_val], y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping]
)
print("Model training completed.")

# Evaluate on test set
print("Evaluating model on test set...")
test_loss = model.evaluate([X_user_test, X_item_test, X_img_test], y_test)
print(f"Test Loss: {test_loss}")

# Function to generate recommendations
def generate_recommendations(model, user_id, n=10):
    print(f"Generating recommendations for user {user_id}...")
    user_index = user_to_index[user_id]
    user_input = np.full(len(item_ids), user_index)
    item_input = np.array(list(item_to_index.values()))
    img_input = np.array([image_features[item_id] for item_id in item_ids])
    
    predictions = model.predict([user_input, item_input, img_input])
    predictions = scaler.inverse_transform(predictions)  # Rescale predictions
    top_indices = predictions.flatten().argsort()[-n:][::-1]
    
    recommendations = [list(item_to_index.keys())[i] for i in top_indices]
    print("Recommendations generated successfully.")
    return recommendations

# Example usage
user_id = ratings_df['user_id'].iloc[0]  # Just using the first user as an example
recommendations = generate_recommendations(model, user_id)
print(f"Top recommendations for user {user_id}:", recommendations)

# Save the model
print("Saving the model...")
model.save('recommendation_model.h5')
print("Model saved successfully.")

print("Recommendation system process completed.")


# In[ ]:




