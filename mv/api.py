import numpy as np
import pickle
from gensim.parsing.porter import PorterStemmer
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import uvicorn
from fastapi import FastAPI
from starlette.responses import HTMLResponse

import middleware.cors
import middleware.logging
from dtos.requests import PredictRequest
from dtos.responses import PredictResponse

from settings import Settings, load_env
from static.render import render
from utilities.utilities import get_uptime

from ml.emily import Emily

emily = Emily()

load_env()

# --- Welcome to your Emily API! --- #
# See the README for guides on how to test it.

# Your API endpoints under http://yourdomain/api/...
# are accessible from any origin by default.
# Make sure to restrict access below to origins you
# trust before deploying your API to production.


app = FastAPI()
settings = Settings()

middleware.logging.setup(app)
middleware.cors.setup(app)


@app.post('/api/predict', response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:

    # You receive all reviews as plaintext in the request.
    # Return a list of predicted ratings between 1-5 (inclusive).
    # You must return the same number of ratings as there are reviews, and each
    # rating will be associated with the review at the same index in the request list.

    # ratings = [random.uniform(0.5, 5.0) for review in request.reviews]
    # ratings = [emily.predict(review) for review in request.reviews]

    # LOAD WORD2VEC MODEL
    model = Word2Vec.load('./word2vec-500.model')

    # LOAD DECISION TREE CLASSIFER
    clf_pkl = open('./decision_tree_classifier.pkl', 'rb')
    clf = pickle.load(clf_pkl)

    # TOKENIZE AND STEM DATA
    porter_stemmer = PorterStemmer()
    tokenized = [simple_preprocess(line, deacc=True)
                 for line in request.reviews]
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
                    print("%s not known" % token)
            if len(sentence_vec) == 0:
                continue

            # SUM VECTORS IN SENTENCE
            sum_vector = (np.mean(sentence_vec, axis=0)).tolist()
            sentence_vectors.append(sum_vector)

    # PREDICT WITH DECISION TREE CLASSIFIER BASED ON SENTENCE VECTOR
    predictions = clf.predict(sentence_vectors).tolist()

    def process(n):
        x = n / 2
        x = int(x)
        return x

    ratings = map(process, predictions)

    return PredictResponse(ratings=list(ratings))


@app.get('/api')
def hello():
    return {
        "uptime": get_uptime(),
        "service": settings.COMPOSE_PROJECT_NAME,

    }


@app.get('/')
def index():
    return HTMLResponse(
        render(
            'static/index.html',
            host=settings.HOST_IP,
            port=settings.CONTAINER_PORT
        )
    )


if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=settings.HOST_IP,
        port=settings.CONTAINER_PORT
    )
