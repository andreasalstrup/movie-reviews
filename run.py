from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
import pandas as pd
from sklearn.model_selection import train_test_split

import pickle
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim.models import Word2Vec

import numpy as np
import pickle
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer

# Edit to fit dataset
CSV_FILE_TRAIN = 'goodshit.csv'
CSV_FILE_TEST = './datasets/testdata-v3.csv'
RATING = 'rating'
TEXT = 'text'
TEST_SIZE = 0.70
############################################################

# LOAD DATA
data = pd.read_csv(r'' + CSV_FILE_TRAIN)

# Remove NaN from dataset
data = data.dropna()

########################################################################################################################
# Split data
#X = data[RATING]
#y = data[TEXT]

# Split arrays or matrices into random train and test subsets
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE, random_state=42)

# shape describes how many dimensions the tensor has along each axis.
#print("Shape")
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

########################################################################################################################

# TOKENIZE AND STEM DATA - TRAIN DATA
porter_stemmer = PorterStemmer()
#tokenized = [simple_preprocess(line, deacc=True) for line in y_train]
tokenized = [simple_preprocess(line, deacc=True) for line in data[TEXT]]
stemmed = [[porter_stemmer.stem(word) for word in tokens]
           for tokens in tokenized]

# TRAIN WORD2VEC
model = Word2Vec(stemmed, min_count=1, vector_size=350,
                 workers=3, window=3, sg=1)

model.save('./word2vec-500.model')

# LOAD WORD2VEC MODEL
model = Word2Vec.load('./word2vec-500.model')

# LOOK UP INDIVIDUAL WORD VECTORS AND SUM THE SENTENCE
sentence_vectors = []
labels = []

for index, sentence in enumerate(stemmed):
    if not len(sentence) == 0:
        model_vector = (np.mean([model.wv[token]
                                 for token in sentence], axis=0)).tolist()

        sentence_vectors.append(model_vector)
        labels.append(data.iat[index, 0])


# TRAIN DECISION TREE CLASSIFIER ON SENTENCE VECTORS USING LABELS
clf = DecisionTreeClassifier()
clf = clf.fit(sentence_vectors, labels)

decision_tree_model_pkl = open('decision_tree_classifier.pkl', 'wb')
pickle.dump(clf, decision_tree_model_pkl)
decision_tree_model_pkl.close()

#################################################################
testData = pd.read_csv(r'' + CSV_FILE_TEST, error_bad_lines=False, engine ='python')
#testData = pd.read_csv(r'' + CSV_FILE_TEST)
sentences = testData[TEXT].tolist()
expected = testData[RATING].tolist()

#sentences = y_test.tolist()
#expected = X_test.tolist()

# LOAD DECISION TREE CLASSIFER
#clf_pkl = open('./decision_tree_classifier.pkl', 'rb')
#clf = pickle.load(clf_pkl)

# TOKENIZE AND STEM DATA - TEST DATA
porter_stemmer = PorterStemmer()
tokenized = [simple_preprocess(line, deacc=True) for line in sentences]
stemmed = [[porter_stemmer.stem(word) for word in tokens]
           for tokens in tokenized]


# CHECK IF PRESENT AND LOOK UP INDIVIDUAL WORD VECTORS
sentence_vectors = []
for index, sentence in enumerate(stemmed):
    if not len(sentence) == 0:
        sentence_vec = []
        for token in sentence:
            try:
                sentence_vec.append(model.wv[token])
            except Exception:
                ""
                #print("%s not known" % token)
        if len(sentence_vec) == 0:
            print('sentence')
            print(sentence)
            sentence_vectors.append(model.wv['movi'])
            continue

        # SUM VECTORS IN SENTENCE
        sum_vector = (np.mean(sentence_vec, axis=0)).tolist()
        sentence_vectors.append(sum_vector)
    else:
        sentence_vectors.append(model.wv['movi'])

# PREDICT WITH DECISION TREE CLASSIFIER BASED ON SENTENCE VECTOR
predictions = clf.predict(sentence_vectors)
#predictions = np.full(2862, 2.5)

print(predictions)
print(len(sentences))
print(len(expected))
print(len(predictions))

err = 0
for index, el in enumerate(expected):
    err += abs(el - predictions[index])

final_err = err / len(expected)

print(final_err)
