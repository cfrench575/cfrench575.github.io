---
title: "Create a hybrid movie recommender using LightFM"
date: 2023-07-13T22:07:32-04:00
#draft: true
---

# Recommendation Algorithms: Background
  Popularized by Amazon in the 1990s, recommendation algorithms have become a staple of business analytics. From shopping add-ons to information sharing via contact the influence of recommendation algorithms is undeniable. Under the hood, recommendation algorithms use both implicit (i.e number of times the item is purchased) and explicit (i.e star ratings) user feedback to determine the strength of association for a recommendation. I want to first define two main types of recommendation algorithms; content-based vs collaborative filtering.

#### Collaborative and Content Filtering
 **Collaborative filtering** uses similar users to recommend new content. For example, a user who historically has enjoyed multiple similar films as you may also have rated highly additional movies that you have not yet discovered - those are the movies that would then be recommended to you.
**Content filtering** on the other hand recommends movies that are similar to movies you have already enjoyed. For example, if you historically rate Western’s highly, the algorithm is likely to recommend other Western movies to you.

#### Matrix Factorization: SVD
 Both of these algorithms rely on a technique called matrix factorization. In practice, datasets for recommendations typically take the format of a user/item (or even an item/item) matrix. Matrix factorization takes advantage of a mathematical property of matrices where any matrix can be broken down (factored) into 3 matrices (noted as Sigma, V and D), and those matrices can then be multiplied together to reconstruct the original matrix. The diagonal of the sorted sigma matrix (called singular values) is then used as weights that represent how important the new dimension (row or column vector) is in expressing the original matrix. For content filtering the dimensions are properties of metadata (items) whereas with collaborative filtering the dimensions are user reactions (ratings). This technique is called **“Singular Value Decomposition”**.
 In practice, recommendation matrices are often quite sparse with many missing values. A modified version of this approach (*simplified SVD*) popularized by Simon Funk in 2009 who used it to win the Netflix prize uses only the V and D matrices to reconstruct the known values of the original matrix - a technique that can account for the high volume of missing values. The values are reconstructed using **gradient descent**. 

 #### Gradient Descent
 Gradient descent tries the “best guess” of weights to minimize error (such as RMSE) and most accurately reconstruct the original matrix. Weights are tested iteratively to minimize error (maximize accuracy) and the model is updated after each new datapoint in the training process.  Simplified SVD with stochastic gradient descent is a powerful and practical method for creating a recommendation algorithm.

#### Data Considerations
With any algorithm, ensuring high quality data is the best way to improve accuracy. For recommendation algorithms, this means choosing powerful variables that are likely to reflect the actual strength of association. For example, when considering movie recommendations, the date of the movie viewing might be relevant; a person’s taste may not be reflected in films last viewed 20 years ago. There are many opportunities for algorithm customization using creative composite/latent engineered features.
Both collaborative and content filtering techniques have their own strengths and weaknesses; collaborative filtering is less effective when there is minimal user data (called the “cold start” problem), and content filtering is limited by the users existing preferences and also may require extensive domain knowledge.
In practice it can be very useful to combine both content and collaborative filtering in a single algorithm as one technique can make up for the others’ shortcomings. LightFM is one such python package that makes it easy to create a hybrid recommendation algorithm that uses both content and user data

# LightFM
 LightFM is a popular python package for hybrid recommendation algorithms that incoporates implicit and explicit feedback. Both item and user metadata are used to calcuate traditional matrix factorization algorithms. Each user and item are represented as the sum of the latent representations of their features, allowing recommendations to generalise to new items (via item features) and to new users (via user features).

 ```python
from pyomo.environ import *
model = Concretemodel()
```

# hybrid-movie-recommender
hybrid movie recommender created using LightFM to generate predictions for friends. Data downloaded from https://www.kaggle.com/rounakbanik/the-movies-dataset?select=movies_metadata and combined with personal movie rating data gathered from friends.


 View the code in the jupyter notebook here: https://nbviewer.jupyter.org/github/cfrench575/hybrid-movie-recommender/blob/main/movieclub_model.ipynb


