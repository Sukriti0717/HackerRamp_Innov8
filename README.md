# HackerRamp_Innov8
Enhancing User Engagement and Personalization through Swipe-Based Product Recommendations

To develop a swipe-based product interaction feature for the Myntra app that allows users to quickly indicate their interest in products by swiping right (interested) or left (not interested). This interaction model aims to improve user engagement, provide immediate and accurate feedback on user preferences, and enhance the recommendation system by incorporating real-time user input. 

# Progress 
For the implementation phase, 
- Successfully implemented the swipe feature in UI, where input is taken from user to analyze their interest.
- Created a simple flask app that allows the user to like or dislike an item, or to simply skip to the next item. 
- We intend to implement the feature that allows user to scroll back to a skipped item, and also to enable them to undo a previous like/dislike.
- We intended to implement a CNN based collaborative filtering model that could be able to identify patterns in the images of products that the users liked or disliked. For simplicity, we have implemented a KNN based model that recommends another product based on current product and user preferences.
- For training the CNN based model, image dataset from Kaggle is used. Due to time constraints, we were unable to fully train this model, but the code is included in our GitHub repository. For now, the recommendation system uses the KNN based model.
- For the KNN model,  FashionDataset.csv dataset has been used, which contains descriptions of the products. The similarity between products is found using the product description and ratings, and TfidfVectorizer is used to convert keywords into tokens.
- Here, the KNN model takes the image_id and the user’s preference as input and suggests a similar product.
- Similarly, in the CNN model’s implementation, the model will take as input the image of the product and how the user interacted with the image, understand patterns and similarities between it, and fetch the recommended product from the database.

# Demo
https://github.com/user-attachments/assets/28267899-f518-4b13-987b-98c3916fe93b

# Benefits
- Enhanced User Engagement: Provides an interactive and fun way for users to browse products, increasing the time spent on the app.
- Modern User Experience: Emulates popular swipe-based interfaces, aligning with current UI/UX trends and user expectations.
- Personalized Recommendations: Tailors product suggestions based on individual user preferences, improving user satisfaction in real-time, increasing the relevance of recommendations.
- Efficient Browsing: Allows users to quickly indicate interest or disinterest in products, making the shopping experience more efficient.
- Demand Forecasting: Accurate demand forecasting based on real-time user interest helps optimize inventory levels, reducing overstock and understock situations and improving supply chain efficiency.
- Increased Sales: By showing users more products they are likely to be interested in, the likelihood of purchase increases and hence increasing conversion rates.
- User Retention: An engaging and personalized shopping experience fosters customer loyalty and encourages repeat visits, driving long-term customer retention and increasing lifetime value.
- Marketing: The detailed preference data collected can be used to create highly targeted marketing campaigns, ensuring that promotional efforts reach the right audience with the right products.

Dataset:
https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset
