# -*- coding: utf-8 -*-

#@title Import Data { display-mode: "form" }
import math
import os
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import GloVe

import pickle

!wget -O data.zip 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Fake%20News%20Detection/inspirit_fake_news_resources%20(1).zip'
!unzip data.zip

basepath = '.'

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
  train_data, val_data = pickle.load(f)

print('Number of train examples:', len(train_data))
print('Number of val examples:', len(val_data))

def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
  if description_tag:
    description = description_tag.get('content') or ''
  else: # If there is no description, return empty string.
    description = ''
  return description

def scrape_description(url):
  if not url.startswith('http'):
    url = 'http://' + url
  response = requests.get(url, timeout=10)
  html = response.text
  description = get_description_from_html(html)
  return description

print('Description of Google.com:')
print(scrape_description('google.com'))



#@title Live Website Description Scraper { display-mode: "both" }

url = "cnn.com" #@param {type:"string"}

print('Description of %s:' % url)
print(scrape_description(url))



def get_descriptions_from_data(data):
  # A dictionary mapping from url to description for the websites in
  # train_data.
  descriptions = []
  for site in tqdm(data):

    descriptions.append(get_description_from_html(site[1]))

  return descriptions


train_descriptions = get_descriptions_from_data(train_data)
train_urls = [url for (url, html, label) in train_data]

print('\nNYTimes Description:')
print(train_descriptions[train_urls.index('nytimes.com')])


val_descriptions = get_descriptions_from_data(val_data)


vectorizer = CountVectorizer(max_features=300)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(descriptions, vectorizer):
  X = vectorizer.transform(descriptions).todense()
  return X

print('\nPreparing train data...')
bow_train_X = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_train_y = [label for url, html, label in train_data]

print('\nPreparing val data...')
bow_val_X = vectorize_data_descriptions(val_descriptions, vectorizer)
bow_val_y = [label for url, html, label in val_data]


model = LogisticRegression()

model.fit(bow_train_X, bow_train_y)
pred = model.predict(bow_train_X)
accuracy = accuracy_score(bow_train_y, pred)
print ("accuracy: ", accuracy)
pred2 = model.predict(bow_val_X)
accuracy2 = accuracy_score(bow_val_y, pred2)

print (confusion_matrix(bow_val_y, pred2))

pfp = precision_recall_fscore_support(bow_val_y, pred2)

print ('prec', pfp[0][1])
print ('recall', pfp[1][1])
print ('f-score', pfp[2][1])


VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

# Returns word vector for word if it exists, else return None.
def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None


good_vector = get_word_vector("good")

print('Shape of good vector:', good_vector.shape)
print(good_vector)

"""### Reading

Not too much to see here–each word vector is a vector of 300 numbers, and it's hard to interpret them from looking at the numbers. Remember that the important property of word vectors is that words with similar meaning have similar word vectors. The magic happens when we compare word vectors.

Below, we have set up a demo where we compare the word vectors for two words using a comparison metric known as cosine similarity. Intuitively, cosine similarity measures the extent to which two vectors point in the same direction. You might be familiar with the fact that the cosine similarity between two vectors is the same as the cosine of the angle between the two vectors–ranging between -1 and 1. -1 means that two vectors are facing opposite directions, 0 means that they are perpindicular, and 1 means that they are facing the same direction.

## Instructor-Led Discussion: Comparing Word Similarities

Try running the below to compare the vectors for "good" and "great", and then try other words, like "planet". **What do you notice that's expected and unexpected? Play around for a couple of minutes then discuss as a class.**

Note that the demo runs automatically when you change either *word1* or *word2*.
"""

#@title Word Similarity { run: "auto", display-mode: "both" }

def cosine_similarity(vec1, vec2):
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

word1 = "plant" #@param {type:"string"}
word2 = "quantum" #@param {type:"string"}

print('Word 1:', word1)
print('Word 2:', word2)

def cosine_similarity_of_words(word1, word2):
  vec1 = get_word_vector(word1)
  vec2 = get_word_vector(word2)

  if vec1 is None:
    print(word1, 'is not a valid word. Try another.')
  if vec2 is None:
    print(word2, 'is not a valid word. Try another.')
  if vec1 is None or vec2 is None:
    return None

  return cosine_similarity(vec1, vec2)


print('\nCosine similarity:', cosine_similarity_of_words(word1, word2))

def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split():
            vec = get_word_vector(word)
            if vec is not None:
                found_words = found_words + 1
                X[i] += vec

        # We divide the sum by the number of words added, so we have the
        # average word vector.
        if found_words > 0:
            X[i] /= found_words

    return X

glove_train_X = glove_transform_data_descriptions(train_descriptions)
glove_train_y = [label for (url, html, label) in train_data]

glove_val_X = glove_transform_data_descriptions(val_descriptions)
glove_val_y = [label for (url, html, label) in val_data]


model = LogisticRegression()

model.fit(glove_train_X, glove_train_y)
pred = model.predict(glove_train_X)
accuracy = accuracy_score(glove_train_y, pred)
print ("accuracy: ", accuracy)
pred2 = model.predict(glove_val_X)
accuracy2 = accuracy_score(glove_val_y, pred2)
print (accuracy2)
print (confusion_matrix(glove_val_y, pred2))

pfp = precision_recall_fscore_support(glove_val_y, pred2)

print ('prec', pfp[0][1])
print ('recall', pfp[1][1])
print ('f-score', pfp[2][1])



def train_model(train_X, train_y, val_X, val_y):
  model = LogisticRegression(solver='liblinear')
  model.fit(train_X, train_y)

  return model


def train_and_evaluate_model(train_X, train_y, val_X, val_y):
  model = train_model(train_X, train_y, val_X, val_y)


  train_y_pred = model.predict(train_X)
  print('Train accuracy', accuracy_score(train_y, train_y_pred))

  val_y_pred = model.predict(val_X)
  print('Val accuracy', accuracy_score(val_y, val_y_pred))

  print('Confusion matrix:')
  print(confusion_matrix(val_y, val_y_pred))

  prf = precision_recall_fscore_support(val_y, val_y_pred)

  print('Precision:', prf[0][1])
  print('Recall:', prf[1][1])
  print('F-Score:', prf[2][1])


  return model


def prepare_data(data, featurizer):
    X = []
    y = []
    for datapoint in data:
        url, html, label = datapoint
        # We convert all text in HTML to lowercase, so <p>Hello.</p> is mapped to
        # <p>hello</p>. This will help us later when we extract features from
        # the HTML, as we will be able to rely on the HTML being lowercase.
        html = html.lower()
        y.append(label)

        features = featurizer(url, html)

        # Gets the keys of the dictionary as descriptions, gets the values
        # as the numerical features. Don't worry about exactly what zip does!
        feature_descriptions, feature_values = zip(*features.items())

        X.append(feature_values)

    return X, y, feature_descriptions

# Gets the log count of a phrase/keyword in HTML (transforming the phrase/keyword
# to lowercase).
def get_normalized_count(html, phrase):
    return math.log(1 + html.count(phrase.lower()))

# Returns a dictionary mapping from plaintext feature descriptions to numerical
# features for a (url, html) pair.
def keyword_featurizer(url, html):
    features = {}

    # Same as before.
    features['.com domain'] = url.endswith('.com')
    features['.org domain'] = url.endswith('.org')
    features['.net domain'] = url.endswith('.net')
    features['.info domain'] = url.endswith('.info')
    features['.org domain'] = url.endswith('.org')
    features['.biz domain'] = url.endswith('.biz')
    features['.ru domain'] = url.endswith('.ru')
    features['.co.uk domain'] = url.endswith('.co.uk')
    features['.co domain'] = url.endswith('.co')
    features['.tv domain'] = url.endswith('.tv')
    features['.news domain'] = url.endswith('.news')

    keywords = ['trump', 'biden', 'clinton', 'sports', 'finance']

    for keyword in keywords:
      features[keyword + ' keyword'] = get_normalized_count(html, keyword)

    return features


train_X, train_y, features = prepare_data(train_data, keyword_featurizer)
val_X, val_y, features = prepare_data(train_data, keyword_featurizer)
train_and_evaluate_model(train_X, train_y, val_X, val_y)


vectorizer = CountVectorizer(max_features=300)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(data_descriptions, vectorizer):
  X = vectorizer.transform(data_descriptions).todense()
  return X

bow_train_X = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_val_X = vectorize_data_descriptions(val_descriptions, vectorizer)

train_and_evaluate_model(bow_train_X, train_y, bow_val_X, val_y)


"""We can see that we are getting similar results, without necessarily using the same information as the keywords-based approach. We then asked whether we could do better by making use of word vectors, which encode the meaning of different words. We found that averaging the word vectors for words in the meta description was a useful approach.

## Exercise

As before, review the approach and complete the code for training and evaluation (~5 minutes).
"""

VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

# Returns word vector for word if it exists, else return None.
def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split():
            vec = get_word_vector(word)
            if vec is not None:
        if found_words > 0:
            X[i] /= found_words

    return X

def combine_features(X_list):
  return np.concatenate(X_list, axis=1)

def get_data_pair(url):
  if not url.startswith('http'):
      url = 'http://' + url
  url_pretty = url
  if url_pretty.startswith('http://'):
      url_pretty = url_pretty[7:]
  if url_pretty.startswith('https://'):
      url_pretty = url_pretty[8:]

  # Scrape website for HTML
  response = requests.get(url, timeout=10)
  htmltext = response.text

  return url_pretty, htmltext

curr_url = "www.yahoo.com" #@param {type:"string"}

url, html = get_data_pair(curr_url)

# Call on the output of *keyword_featurizer* or something similar
# to transform it into a format that allows for concatenation. See
# example below.
def dict_to_features(features_dict):
  X = np.array(list(features_dict.values())).astype('float')
  X = X[np.newaxis, :]
  return X
def featurize_data_pair(url, html):
  # Approach 1.
  keyword_X = dict_to_features(keyword_featurizer(url, html))
  # Approach 2.
  description = get_description_from_html(html)

  bow_X = vectorize_data_descriptions([description], vectorizer)

  # Approach 3.
  glove_X = glove_transform_data_descriptions([description])

  X = combine_features([keyword_X, bow_X, glove_X])

  return X

curr_X = featurize_data_pair(url, html)

model = train_model(combined_train_X, train_y, combined_val_X, val_y)

curr_y = model.predict(curr_X)[0]


if curr_y < .5:
  print(curr_url, 'appears to be real.')
else:
  print(curr_url, 'appears to be fake.')
