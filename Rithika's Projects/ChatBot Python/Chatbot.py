import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# install nltk

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages
nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only

f=open(r'C:\Users\USER\Desktop\rithika.txt',errors = 'ignore')
raw=f.read()
raw = raw.lower()# converts to lowercase
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words
lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
GREETING_INPUTS = ("hello", "hi", "greetings", "ssup","hola","namasthe", "wassup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello!!", "Hi! nice to meet you."]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(ur_response):
    r_response=''
    sent_tokens.append(ur_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        r_response=r_response+"I am sorry! I don't understand you :("
        return r_response
    else:
        r_response = r_response+sent_tokens[idx]
        return r_response
flag=True
print("ChatBot: My name is Rithika. I will answer your queries about XXX. If you want to exit, type Bye!")
while(flag==True):
    ur_response = input()
    ur_response=ur_response.lower()
    if(ur_response!='bye'):
        if(ur_response=='thanks' or ur_response=='thank you' ):
            flag=False
            print("Rithika: You are welcome..")
        else:
            if(greeting(ur_response)!=None):
                print("Rithika: "+greeting(ur_response))
            else:
                print("Rithika: ",end="")
                print(response(ur_response))
                sent_tokens.remove(ur_response)
    else:
        flag=False
        print("Rithika : Bye! take care..")

