from bertopic.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TfIdfEmbedder(BaseEmbedder):
    def __init__(self, vocabulary: list[str] | None = None, min_df=5, stop_words='english'):
        super().__init__()
        self.embedding_model: TfidfVectorizer = TfidfVectorizer(min_df=min_df, stop_words=stop_words)
        self.vocabulary = vocabulary
        if vocabulary is not None:
            self.embedding_model.fit(self.vocabulary)

    def embed(self, documents, verbose=False):
        if self.vocabulary is None:
            return self.embedding_model.fit_transform(documents)
        return self.embedding_model.transform(documents)


class CountVectorizerEmbedder(BaseEmbedder):
    def __init__(self, vocabulary: list[str] | None = None, min_df=5, n_gram_range=(1, 3)):
        super().__init__()
        self.embedding_model: CountVectorizer = CountVectorizer(min_df=min_df, ngram_range=n_gram_range)
        self.vocabulary = vocabulary
        if vocabulary is not None:
            self.embedding_model.fit(self.vocabulary)

    def embed(self, documents, verbose=False):
        if self.vocabulary is None:
            return self.embedding_model.fit_transform(documents)
        return self.embedding_model.transform(documents)


class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed(self, documents, verbose=False):
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings
