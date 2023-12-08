---
title: "Create a Hybrid Movie Recommender Using LightFM With Python"
date: 2023-07-13T22:07:32-04:00
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

# LightFM movie recommender in Python
 LightFM is a popular python package for hybrid recommendation algorithms that incoporates implicit and explicit feedback. Both item and user metadata are used to calcuate traditional matrix factorization algorithms. Each user and item are represented as the sum of the latent representations of their features, allowing recommendations to generalise to new items (via item features) and to new users (via user features). 
 
 The following is a brief demo of the LightFM package that uses plot keywords, ratings and movie metadata from https://www.kaggle.com/rounakbanik/the-movies-dataset?select=movies_metadata which I combined with personal movie rating data gathered from friends (gathering data from your friends is completely normal).
 
 #### Load Data
 ```python
from math import sqrt

import pandas as pd

import json
import numpy as np

from google.colab import drive
drive.mount('/content/drive/')

# Pandas config
def pandas_config():
    # display 10 rows and all the columns
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', None)

    
pandas_config()
```

```python
ratings = pd.read_csv('/content/drive/MyDrive/online learning/movierecs/data/ratings.csv')
movieclub = pd.read_csv('/content/drive/MyDrive/online learning/movierecs/data/movieclub.csv')
keywords = pd.read_csv('/content/drive/MyDrive/online learning/movierecs/data/keywords.csv')
movie_metadata = pd.read_csv('/content/drive/MyDrive/online learning/movierecs/data/movies_metadata.csv')
```

#### Data Preparation
Prepare ratings, movieclub and movies data to be merged.

```python
ratings = ratings[["userId", "movieId", "rating"]]
movieclub = movieclub[["userId", "movieId", "rating"]].dropna() 
movieclub['movieId'] = movieclub['movieId'].astype(int)
movieclub.head()

movies = movie_metadata[['id', 'title']].dropna() 
movies = movies.rename(columns={'id': 'movieId'})
movies['movieId'] = movies['movieId'].astype(int)
movies.head()
```
Reformat dictionary keyword data so that each movie has a array of keywords. Movie keywords will be used for the content-based part of the recommendation algorithm.

```python
column_of_lists=[]
for i in range(len(keywords)): 
    row = keywords.loc[i, "keywords"]
    try:
        dictionaries = json.loads(row.replace("'", "\""))
        keyword_list=[]
        for Dict in dictionaries:
            keyword=(Dict['name'])
            keyword_list.append(keyword)
    except:
        keyword_list.append([])
    column_of_lists.append(keyword_list)
    
keywords['word_array']=column_of_lists
keywords['id']=keywords['id'].astype(int)

keywords = keywords.rename(columns={'id': 'movieId'})
keywords = keywords[['movieId', 'word_array']]
keywords.head()
```
Concatonate MovieLens data with friends' rating data. The movieLens data is very large, so to improve latency I'm filtering out movies that we haven't watched as a group. The resulting dataset consists of user ratings (both from the MovieLens dataset and my friends) from movies that we have watched together.

```python
### Alissa    270897
### Byron     270898
### Chelsea   270899
### Hannah    270900
### Harrison  270901
### Martin    270902
### Michael   270903
### Katherine 270904

ratings = pd.concat([ratings, movieclub], ignore_index=True)
ratings.reset_index(drop=True, inplace=True)

movies_watched = list(movieclub.movieId.unique())
movies_to_keep = movies_watched + [480]

movies = movies[movies.movieId.isin(movies_to_keep)]
movies.head()
```
Also filter out unwatched movies from the keywords dataset.

```python
keywords['word_array'] = keywords['word_array'].astype('str') 
keywords= keywords.replace('\[','', regex=True)
keywords= keywords.replace('\]','', regex=True)

keywords = keywords[keywords.movieId.isin(movies_to_keep)]
```
Merge ratings, movies and keywords into a single dataframe.

```python
ratings2= pd.merge(ratings, movies, on="movieId")
ratings2.head()

data = pd.merge(ratings2, keywords, on="movieId")
data['movieId'] = data['movieId'].astype('str') 
```
#### LightFM Data Matricies
The lightFM algorithm takes up to three inputs: the user-item interaction matrix, the item feature matrix and the user feature matrix.

* User-item interaction matrix
```python
#### user-item interaction matrix (movie ratings)
df=data[["movieId", "userId", "rating"]].reset_index()
interactions = df.groupby(['userId', 'movieId'])['rating'].sum().unstack().fillna(0)
interactions.tail(10)
```

* Also generate dictionaries for user and item names and ids
```python
### user id/index dictionary for accessing user data for predictions 
user_id = list(interactions.index)
user_dict = {}
counter = 0
for i in user_id:
    user_dict[i] = counter
    counter += 1

### item dictionary (movie id/ title)
item_dict = {}
for i in range(data.shape[0]):
    item_dict[(data.loc[i,"movieId"])] = data.loc[i,"title"]

```

* Item-feature matrix
Plot keywords are reformatted into a sparse matrix. This matrix will be used to compute latent variables for the content-based part of the recommendation algorithm.
```python
def tokens(x):
    return x.split(', ')
                                                                                 
item_features = data[["movieId", "word_array"]].drop_duplicates(["movieId", "word_array"]).reset_index(drop=True)
                                                                  
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(tokenizer=tokens, max_features = 100)

item_features_csr = cv.fit_transform(item_features['word_array'])
```

#### LightFM Model

```python
### imports
!pip install lightfm
from lightfm import LightFM
import scipy
from scipy import sparse 
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
```
LightFM accepts following algorithms as an argument:
* logistic: useful when both positive (1) and negative (-1) interactions are present.
* BPR: Bayesian Personalised Ranking pairwise loss. Maximises the prediction difference between a positive example and a randomly chosen negative example. Useful when only positive interactions are present and optimising ROC AUC is desired.
* WARP: Weighted Approximate-Rank Pairwise loss. Maximises the rank of positive examples by repeatedly sampling negative examples until rank violating one is found. Useful when only positive interactions are present and optimising the top of the recommendation list (precision@k) is desired.
* k-OS WARP: k-th order statistic loss. A modification of WARP that uses the k-th positive example for any given user as a basis for pairwise updates.

```python
x = sparse.csr_matrix(interactions.values)
train, test = random_train_test_split(interactions=x, test_percentage=0.2)
model = LightFM(no_components= 50, loss='warp')
model.fit(x,epochs=10, item_features=item_features_csr, user_features=None)
```

#### LightFM Model Evaluation
Evaluations metrics (A perfect score is 1.0.): 
* AUC: the probability that a randomly chosen positive example has a higher score than a randomly chosen negative example.
* Precision at k: the fraction of known positives in the first k positions of the ranked list of results.
* Recall at k: the number of positive items in the first k positions of the ranked list of results divided by the number of positive items in the test period.
* Reciprocal rank: 1 / the rank of the highest ranked positive example.

```python
train_precision = precision_at_k(model, train, k=5, item_features=item_features_csr).mean()
test_precision = precision_at_k(model, test, k=5, item_features=item_features_csr).mean()

train_auc = auc_score(model, train, item_features=item_features_csr).mean()
test_auc = auc_score(model, test, item_features=item_features_csr).mean()

print('Precision: train %.2f' % (train_precision))
print('AUC: train %.2f' % (train_auc))
print('Precision: test %.2f' % (test_precision))
print('AUC: test %.2f' % (test_auc))
```


Lastly, save the model and dictionaries for future use to predict your friend's reactions to movies you will watch together in the future. 

```python
import pickle
import json
from scipy import sparse
import numpy
pickle.dump(model, open('model.pkl','wb'))

def convert(o):
    if isinstance(o, numpy.int64): return int(o)  
    raise TypeError

#Save user ratings matrix
sparse.save_npz('rating_interaction.npz', x)

### save item feature sparse matrix
sparse.save_npz('item_feature_sparse.npz', item_features_csr)

pickle.dump(interactions, open('interactions.pkl','wb'))

## save and reload item embeddings
item_representations=model.get_item_representations()
pickle.dump(item_representations, open('item_representations.pkl','wb'))

item_name_dict = cv.vocabulary_

#### save dictionaries
with open('item_name_dict.json', 'w') as fp:
    json.dump(item_name_dict, fp, default=convert)

with open('item_dict.json', 'w') as fp:
    json.dump(item_dict, fp, default=convert)

with open('user_dict.json', 'w') as fp:
    json.dump(user_dict, fp, default=convert)
```

 View jupyter notebook for this analysis here: https://nbviewer.jupyter.org/github/cfrench575/hybrid-movie-recommender/blob/main/movieclub_model.ipynb


