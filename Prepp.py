#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.stem.porter import *
stemmer = PorterStemmer()
import pickle
with open('tokenizer.pickle' , 'rb') as handle:
    tokenizer = pickle.load(handle)
from keras.preprocessing.sequence import pad_sequences 

class prep:
    maxlen = 120
    def remove_punc(st):
        s = st.replace("[^a-zA-Z#]", " ")
        return s

    def remove_small(st):

        return ' '.join([w for w in st.split() if len(w)>3])

    def stemming(st):
        text = []
        stem_text =  [stemmer.stem(i) for i in st]
        text.append(''.join(i for i in stem_text))
        return text

    def tokenize( st , Tokenizer = tokenizer ):
        return Tokenizer.texts_to_sequences(st)
    def pad_sequence(string , maxlen = maxlen):
        return pad_sequences(string , maxlen)


# In[ ]:




