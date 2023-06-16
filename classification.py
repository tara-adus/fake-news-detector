@title Live Fake News Classification Demo { run: "auto", vertical-output: true, display-mode: "both" }
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
