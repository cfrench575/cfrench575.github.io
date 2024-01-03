---
title: "Association Rule Learning Using Mlxtend with Python"
date: 2023-12-07T23:58:55-08:00
tags:
- Association rules
- Apriori Analysis
- Data Mining
- Machine Learning
- Python
- Recommendation Algorithms
metaAlignment: center
thumbnailImagePosition: "right"
thumbnailImage: https://img.freepik.com/free-vector/smart-retail-abstract-concept-illustration_335657-3715.jpg?
# Image by vectorjuice on Freepik
---

# Association Rule Learning: Background
Association Rule Learning (also called **Market Basket Analysis**) is a practical and highly interpertable starting place for implementing the first recommendation algorithm for your business. Associaiton rules, or strong relationships between variables in a dataset, can be mined from historical data using an appropriate algorithm. Those rules can then be leveraged to effectively predict future user behavior. Association rules are commonly applied to assist with marketing decisions such as selecting users for a specific ad campaign, recommending personalized services, or smart product up-selling at checkout. 

{{< html >}}
<br>
{{< /html >}}

- [Association Rule Learning: Background](#association-rule-learning-background)
    - [Applications](#applications)
    - [Apriori Algorithm](#apriori-algorithm)
    - [Data Considerations](#data-considerations)
- [Association Rule Recommender in Python using Mlxtend Python Package](#association-rule-recommender-in-python-using-mlxtend-python-package)
    - [Load Data](#load-data)
    - [Data Preparation](data-preparation)
        - [One Hot Encoding](#one-hot-encoding)
    - [Compute Frequent Item Sets using Ariori Algorithm](#compute-frequent-item-sets-using-ariori-algorithm)
    - [Compute Association Metrics from Frequent Item Sets](#compute-association-metrics-from-frequent-item-sets)
    - [Examine Association Rules](#examine-association-rules)
- [Final Thoughts](#final-thoughts)
    - [Networks of Association](#networks-of-association)
    - [Further Applications](#further-applications)

#### Applications
While rules can be hard-coded using prerequisite knowledge, market basket analysis allows for automation which saves time and rules can adapt rapidly to changing customer bases. Additionally, algorithmically mining for rules doesn't require expert domain knowledge, as theoretically the necessary information should already exist in the data. **Market basket analysis** also enables the discovery of highly predictive, less intuitive rules that might otherwise be overlooked by a human; sometimes items frequently purchased together don't always seem similar enough to recommend together, for example, diapers and beer. 

#### Apriori Algorithm
The Apriori algorithm is a type of association rule machine learning. Frequent itemsets are discovered iteratively using breadth-first search. Rules are then calculated via **association metrics** describing the liklihood of the presence of item 2 given item 1. Lastly, unnecesary or redundant rules are pruned. Association metrics like **support, confidence, lift** and **conviction** at user-defined thresholds are calculated to determine the strength and usefulness of the rule. Rules can be calculated live, or exported to a database for later use in email campaigns or recommendation algorithms. 

#### Data Considerations
Typically data for mining association rules is in list format. Each row is a transaction and each transaction contains a list of items. How to define a "transaction" can change the meaning and application of the rules; for some scenarios it might be best to define a transaction as items purchased together at the same time whereas for other scenarios it could be more beneficial to consider a single transaction as the total items purchased by a user in the past 6 months, or even services requested from the past year. Mlxtend is an intuitive python package for mining and saving association rules that accepts input data in a list format. 

# Association Rule Recommender in Python using Mlxtend Python Package 
**Mlxtend** is a popular python package for performing market basket analysis. The following is an example of how to use mlxtend to calculate support, confidence, lift and conviction from shopping cart transaction data downloaded from Kaggle https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset/

#### Load Data

```python
from math import sqrt
import pandas as pd
import json
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from google.colab import drive
drive.mount('/content/drive/')

# Pandas config
def pandas_config():
    # display 10 rows and all the columns
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', None)

pandas_config()

data = pd.read_csv('/content/drive/MyDrive/online learning/marketbasket/Retail_Transactions_Dataset.csv')
```
#### Data Preparation
Take the "Product" column and remove quotes and brackets. Then split each row on the comma so the transactions are in list format. 

```python
data.dtypes
data["Product"] = data["Product"].str.replace(r"'", "")
data["Product"] = data["Product"].str.replace(r"^\[|\]$", "")

# Split transaction strings into lists
transactions = data['Product'].apply(lambda t: t.split(', '))

# Convert DataFrame column into list of strings
transactions = list(transactions)

# Print the list of transactions
print(transactions)
```
##### One Hot Encoding
Transform list transaction data using one hot encoding (coding for categorical variables). 

```python
# Instantiate transaction encoder and identify unique items
encoder = TransactionEncoder().fit(transactions)

# One-hot encode transactions
onehot = encoder.transform(transactions)

# Convert one-hot encoded data to DataFrame
onehot = pd.DataFrame(onehot, columns = encoder.columns_)

# Print the one-hot encoded transaction dataset
print(onehot)
```
#### Compute Frequent Item Sets using Ariori Algorithm 
Use one hot encoded transaction data to compute frequent item sets. Select a minimum suport threshold for determining frequent item sets; the minimum support threshold is a function of the number of times that items occur within the same transaction and depends on the total number of transacitons in you data. Redundant or otherwise unnecessary item sets are dropped. 

```python 
# Compute frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(onehot,
                            min_support = 0.001,
                            max_len = 100,
                            use_colnames = True)


frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets[(frequent_itemsets["length"] < 5)]
```
#### Compute Association Metrics from Frequent Item Sets
Generate association rules from frequent item sets and association statistics. 

```python
# Compute all association rules for frequent_itemsets
rules = association_rules(frequent_itemsets[["support", "itemsets"]],
                            metric = "lift")
# Print association rules
print(rules.shape)

## reformat rules from frozen set to string
# convert to list
rules['antecedent']=rules['antecedents'].apply(list)
rules['consequent']=rules['consequents'].apply(list)
# convert to string
rules['antecedent']=rules['antecedent'].apply(str)
rules['consequent']=rules['consequent'].apply(str)
# remove brackets
rules=rules.replace(']', '', regex=True)
rules=rules.replace('\\[', '', regex=True)
rules=rules.replace('\'', '', regex=True)
# print first 5 rows of rules data
rules.head()
```
Association metrics are computed to describe the strenght/importance of the rule. A single metric in isolation can be misleading so it is good practice to evaualate the association rule using multiple metrics. Below is a brief explanation of each association metric. 

* **Support**
    * Number of transactions with shared items / total number of transactions
    * Frequently purchased items tend to have the highest support
* **Confidence**
    * Support of items X and Y / support of X
    * Captures probability that Y will be purchased is X is purchased; How does the proabaility of purchasing Y change once X is purchased. 
* **Lift**
    * (Support of items X and Y) / (support of X * support of Y)
    * The proportion of transactions that contain X and Y divided by the proportion of transactions if X and Y were randomly and independently assigned. 
    * Evaluates the associations between items excluding association due to random chance
* **Conviction**
    * (Support X and Y) - (Support of X * Support of Y)
    * Similar to lift, but bounded between -1 and +1
* **Zhang’s metric**
    * (Degree of association - dissociation)  / max of two confidence measures (confidence of X then Y, confidence of not X then Y)
    * Dissociation metric bound between -1 and +1 with +1 indicating a perfect association

#### Examine Association Rules 
Visualize metrics using a heatmap.
```python
# # Replace frozen sets with strings
rules['antecedents'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequents'] = rules['consequents'].apply(lambda a: ','.join(list(a)))

# Transform data to matrix format and generate heatmap
pivot = rules.pivot(index='consequents', columns='antecedents', values='support')
sns.heatmap(pivot)
```
Query rules and view recommendations for items of interest.
```python
rules[(rules["antecedent"] == 'Bread') & (rules["leverage"] > 0.0001)].sort_values(by='consequent support', ascending=False)
rules[(rules["antecedent"] == 'Trash Bags') & (rules["leverage"] > 0.0001)].sort_values(by='consequent support', ascending=False)
rules[(rules["antecedent"] == 'Air Freshener') & (rules["leverage"] > 0.0001)].sort_values(by='consequent support', ascending=False)
rules[(rules["antecedent"] == 'Bath Towels') & (rules["leverage"] > 0.0001)].sort_values(by='consequent support', ascending=False)
rules[(rules["antecedent"] == 'Sponges') & (rules["leverage"] > 0.0001)].sort_values(by='consequent support', ascending=False)
rules[(rules["antecedent"] == 'Soap') & (rules["leverage"] > 0.0001)].sort_values(by='consequent support', ascending=False)
rules[(rules["antecedent"] == 'Yogurt') & (rules["leverage"] > 0.0001)].sort_values(by='consequent support', ascending=False)
rules[(rules["antecedent"] == 'Dustpan') & (rules["leverage"] > 0.00001)].sort_values(by='support', ascending=False)
```
# Final Thoughts

After computation, association rules can be visualized as scatter plots using several association metrics (for example, convidence vs support) - this provides a quick and intuitive way to evaluate the usefulness of some of the rules and can assist with pruning. Once you have a dataframe with useful, interpertable, and strong associations between items, that data can be stored in a database connected to your web app and you are ready to start rolling out smart recommendations to users. Items can be recommended based on what the user currently has in their cart, or even their past browsing or purchasing behavior within a specified time frame. Dimensionality reduction techniques can be applied to add an element of randomness to the recommendations and give less popular or less searched items more visibility to the users most likely to purchase them. 

##### Networks of Association
Association analysis can also detect rules with multiple antecedents and more complicated association patterns to generate recommendations based on sequences of items purchased or viewed. Coordinate plots are a useful way to visualize a more complicated network of rules. 

##### Further Applications
Association analyses have a wide range of application beyond just e-commerce and shopping described in this example - for example, market basked analysis can provide medical diagnoses based on clusters of reported patient symptoms, used to gain insight into user behavior on a web application based on networks of pages viewed, or can even be applied to credit card fraud detection to identify common sequences of credit card transactions associated with fradulent behavior. 



