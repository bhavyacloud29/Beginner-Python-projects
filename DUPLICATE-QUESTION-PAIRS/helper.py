import re
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import distance
from fuzzywuzzy import fuzz
import pickle
import numpy as np 

cv=pickle.load(open('cv.pkl','rb'))

def preprocess(q):
  q=str(q).lower().strip()

  q=q.replace('%','percent')
  q=q.replace('$','dollar')
  q=q.replace('₹','rupee')
  q=q.replace('€','euro')
  q=q.replace('@',"at")

  q=q.replace('[math]','')
  q=q.replace(',000,000,000','b ')
  q=q.replace(',000,000','m ')
  q=q.replace(',000','k ')
  q=re.sub(r'([0-9]+)000000000',r'\1b',q)
  q=re.sub(r'([0-9]+)000000',r'\1m',q)
  q=re.sub(r'([0-9]+)000',r'\1k',q)

  #decontracting words
  # Source - https://stackoverflow.com/a/19794953
# Posted by arturomp, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-02, License - CC BY-SA 3.0

  contractions = {
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
  q_contracted=[]
  for word in q.split():
    if word in contractions:
      word=contractions[word]

    q_contracted.append(word)
  q=' '.join(q_contracted)
  q=q.replace("'ve","have")
  q=q.replace("'re","are")
  q=q.replace("'ll","will")
  q=q.replace("n't","not")

  #Remove HTML tags
  q=BeautifulSoup(q)
  q=q.get_text()

  #Remove punctuations
  pattern=re.compile('\W')

  q=re.sub(pattern,' ',q).strip()
  return q

def common_words(q1_str, q2_str):
  w1=set(map(lambda word:word.lower().strip(),q1_str.split(" ")))
  w2=set(map(lambda word:word.lower().strip(),q2_str.split(" ")))
  return len(w1&w2)

def total_words(q1_str, q2_str):
  w1=set(map(lambda word:word.lower().strip(), q1_str.split(" ")))
  w2=set(map(lambda word:word.lower().strip(), q2_str.split(" ")))
  return (len(w1)+len(w2))


from nltk.corpus import stopwords
def fetch_token_features(q1, q2):
  safe_div=0.0001
  stop_words=stopwords.words('english')
  token_features=[0.0]*8

  q1_tokens=q1.split()
  q2_tokens=q2.split()

  if len(q1_tokens)==0 or len(q2_tokens)==0:
    return token_features

  q1_words=set([word for word in q1_tokens if word not in stop_words])
  q2_words=set([word for word in q2_tokens if word not in stop_words])

  q1_stops=set([word for word in q1_tokens if word in stop_words])
  q2_stops=set([word for word in q2_tokens if word in stop_words])

#Get common non-stopwords from question pair
  common_word_count=len(q1_words.intersection(q2_words))
  common_stop_count=len(q1_stops.intersection(q2_stops))

  token_features[0]=common_word_count/(min(len(q1_words),len(q2_words))+safe_div)
  token_features[1]=common_word_count/(max(len(q1_words),len(q2_words))+safe_div)
  token_features[2]=common_stop_count/(min(len(q1_stops),len(q2_stops))+safe_div)
  token_features[3]=common_stop_count/(max(len(q1_stops),len(q2_stops))+safe_div)
  token_features[4]=common_stop_count/(min(len(q1_tokens),len(q2_tokens))+safe_div)
  token_features[5]=common_stop_count/(max(len(q1_tokens),len(q2_tokens))+safe_div)

  #Last word of both questions is same or not
  token_features[6]=int(q1_tokens[-1]==q2_tokens[-1])

  #First word of both questions is same or not
  token_features[7]=int(q1_tokens[0]==q2_tokens[0])

  return token_features


def fetch_length_features(q1, q2):
  length_features=[0.0]*3
  # converting the sentence into tokens
  q1_tokens=q1.split()
  q2_tokens=q2.split()
  if len(q1_tokens)==0 or len(q2_tokens)==0:
    return length_features
  length_features[0]=abs(len(q1_tokens)-len(q2_tokens))

  #avg length of tokens of both questions

  length_features[1]=((len(q1_tokens)+len(q2_tokens))/2)

  str_lcs=list(distance.lcsubstrings(q1,q2))
  length_features[2]=len(str_lcs[0])/(min(len(q1),len(q2))+1)

  return length_features


from fuzzywuzzy import fuzz

def fetch_fuzzy_features(q1, q2):
  fuzzy_features=[0.0]*4
  fuzzy_features[0]=fuzz.ratio(q1,q2)

  fuzzy_features[1]=fuzz.partial_ratio(q1,q2)

  fuzzy_features[2]=fuzz.token_sort_ratio(q1,q2)

  fuzzy_features[3]=fuzz.token_set_ratio(q1,q2)

  return fuzzy_features

def query_point_creator(q1,q2):
  input_query=[]

  q1=preprocess(q1)
  q2=preprocess(q2)

  # Fetch basic features

  input_query.append(len(q1))
  input_query.append(len(q2))

  input_query.append(len(q1.split(" ")))
  input_query.append(len(q2.split(" ")))

  input_query.append(common_words(q1,q2))
  input_query.append(total_words(q1,q2))
  input_query.append(round(common_words(q1,q2)/total_words(q1,q2),2)) # Corrected round function call

  # Fetch token features
  token_features=fetch_token_features(q1,q2)
  input_query.extend(token_features)

  # Fetch length features

  length_features=fetch_length_features(q1,q2)
  input_query.extend(length_features)

  # Fetch fuzzy feature

  fuzzy_features=fetch_fuzzy_features(q1,q2)
  input_query.extend(fuzzy_features)

  # Vectorize the questions

  q1_bow=cv.transform([q1]).toarray()
  q2_bow=cv.transform([q2]).toarray()


  return np.hstack((np.array(input_query).reshape(1,22),q1_bow,q2_bow))

