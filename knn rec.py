#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("D:\\myntradata\\FashionDataset.csv")

# 1. Data preprocessing
df.dropna(inplace=True)
df = df.drop(['name'], axis=1)  # Dropping 'name' column as it's repeated in description

# 2. Convert description and p_attributes to tokens
df['combined_text'] = df['description'] + ' ' + df['p_attributes']
vectorizer = TfidfVectorizer(stop_words='english')
text_matrix = vectorizer.fit_transform(df['combined_text'])

# 3. Prepare numerical features
numerical_features = ['price', 'ratingCount', 'avg_rating']
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[numerical_features])

# Combine text features and numerical features
combined_features = np.hstack((text_matrix.toarray(), scaled_features))

# Calculate cosine similarity
similarity_matrix = cosine_similarity(combined_features)

def recommend_item(image_id, user_preference, k=5):
    # Find the index of the input image_id
    item_index = df[df['p_id'] == image_id].index[0]
    
    # Get similarity scores for the item
    item_scores = list(enumerate(similarity_matrix[item_index]))
    
    # Sort the scores in descending order
    sorted_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)
    
    # Remove the input item itself
    sorted_scores = [score for score in sorted_scores if score[0] != item_index]
    
    if user_preference == 1:
        # User liked the item, recommend the next best item
        recommended_index = sorted_scores[0][0]
    elif user_preference == 0:
        # User is neutral, recommend the k-1 th item
        recommended_index = sorted_scores[k-2][0]
    else:  # user_preference == -1
        # User disliked the item, recommend the least similar item
        recommended_index = sorted_scores[-1][0]
    
    recommended_item = df.iloc[recommended_index]
    return recommended_item



# In[12]:


# Example usage
image_id = 9867983  # Replace with actual image_id
user_preference = 1  # 1 for like, 0 for neutral, -1 for dislike

recommended_item = recommend_item(image_id, user_preference)
print(f"Recommended item:\n{recommended_item[['p_id', 'price', 'colour', 'brand']]}")
print(recommended_item['img'])


# In[13]:


image_id = 18156520  # Replace with actual image_id
user_preference = 1  # 1 for like, 0 for neutral, -1 for dislike

recommended_item = recommend_item(image_id, user_preference)
print(f"Recommended item:\n{recommended_item[['p_id', 'price', 'colour', 'brand']]}")
print(recommended_item['img'])


# In[14]:


image_id = 15636298  # Replace with actual image_id
user_preference = 0  # 1 for like, 0 for neutral, -1 for dislike

recommended_item = recommend_item(image_id, user_preference)
print(f"Recommended item:\n{recommended_item[['p_id', 'price', 'colour', 'brand']]}")
print(recommended_item['img'])

