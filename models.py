import os
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


class Model:
    def load_embeddings(self, train_data1, train_data2):
        raise NotImplementedError

    def classify(self, test_data, embeddings, category1, category2):
        raise NotImplementedError


class BAAIModel(Model):
    def load_embeddings(self, train_data1, train_data2):
        model = SentenceTransformer("BAAI/bge-large-en")
        embeddings1 = model.encode(train_data1, normalize_embeddings=True)
        embeddings2 = model.encode(train_data2, normalize_embeddings=True)

        # Compute average embeddings
        avg_embedding1 = np.mean(embeddings1, axis=0)
        avg_embedding2 = np.mean(embeddings2, axis=0)

        return avg_embedding1, avg_embedding2

    def classify(self, test_data, embeddings, category1, category2):
        model = SentenceTransformer("BAAI/bge-large-en")
        results = []
        for sentence in test_data:
            sentence_embedding = model.encode([sentence], normalize_embeddings=True)[0]
            similarities = cosine_similarity([sentence_embedding], embeddings)
            if similarities[0][0] > similarities[0][1]:
                results.append(category1)
            else:
                results.append(category2)
        return results


class GloveModel(Model):
    def load_embeddings(self, train_data1, train_data2):
        model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d"
        )
        embedding1 = model.encode([" ".join(train_data1)])[0]
        embedding2 = model.encode([" ".join(train_data2)])[0]
        return embedding1, embedding2

    def classify(self, test_data, embeddings, category1, category2):
        model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d"
        )
        results = []
        for sentence in test_data:
            sentence_embedding = model.encode([sentence])[0]
            cosine_to_cat1 = cosine_similarity([sentence_embedding], [embeddings[0]])
            cosine_to_cat2 = cosine_similarity([sentence_embedding], [embeddings[1]])
            if cosine_to_cat1 > cosine_to_cat2:
                results.append(category1)
            else:
                results.append(category2)
        return results


class TFIDFModel(Model):
    def load_embeddings(self, train_data1, train_data2):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(train_data1 + train_data2)
        return vectorizer, tfidf_matrix, len(train_data1), len(train_data2)

    def classify(self, test_data, embeddings, category1, category2):
        vectorizer, tfidf_matrix_train, len_train_data1, _ = embeddings

        tfidf_matrix_test = vectorizer.transform(test_data)

        results = []
        for index, sentence in enumerate(test_data):
            cosine_to_cat1 = cosine_similarity(
                tfidf_matrix_test[index], tfidf_matrix_train[:len_train_data1]
            ).mean()
            cosine_to_cat2 = cosine_similarity(
                tfidf_matrix_test[index], tfidf_matrix_train[len_train_data1:]
            ).mean()

            if cosine_to_cat1 > cosine_to_cat2:
                results.append(category1)
            else:
                results.append(category2)
        return results


class BAAIModel_FineTuned(Model):
    def load_embeddings(self, train_data1, train_data2):
        model = SentenceTransformer("exp_finetune_entire_dataset")
        embeddings1 = model.encode(train_data1, normalize_embeddings=True)
        embeddings2 = model.encode(train_data2, normalize_embeddings=True)

        # Compute average embeddings
        avg_embedding1 = np.mean(embeddings1, axis=0)
        avg_embedding2 = np.mean(embeddings2, axis=0)

        return avg_embedding1, avg_embedding2

    def classify(self, test_data, embeddings, category1, category2):
        model = SentenceTransformer("exp_finetune_entire_dataset")
        results = []
        for sentence in test_data:
            sentence_embedding = model.encode([sentence], normalize_embeddings=True)[0]
            similarities = cosine_similarity([sentence_embedding], embeddings)
            if similarities[0][0] > similarities[0][1]:
                results.append(category1)
            else:
                results.append(category2)
        return results


class OpenAI_AdaModel(Model):
    def __init__(self):
        # Load API key when model is instantiated
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key

    def get_embedding(self, text_to_embed):
        # Limiting character length to 32k
        if len(text_to_embed) > 32000:
            text_to_embed = text_to_embed[:32000]

        response = openai.Embedding.create(
            model="text-embedding-ada-002", input=[text_to_embed]
        )
        embedding = response["data"][0]["embedding"]
        return embedding

    def load_embeddings(self, train_data1, train_data2):
        embeddings1 = [
            self.get_embedding(text) for text in train_data1 if len(text) <= 32000
        ]
        embeddings2 = [
            self.get_embedding(text) for text in train_data2 if len(text) <= 32000
        ]

        # Convert embeddings to numpy array for operations
        embeddings1 = np.array(embeddings1)
        embeddings2 = np.array(embeddings2)

        # Compute average embeddings
        avg_embedding1 = np.mean(embeddings1, axis=0)
        avg_embedding2 = np.mean(embeddings2, axis=0)

        return avg_embedding1, avg_embedding2

    def classify(self, test_data, embeddings, category1, category2):
        results = []
        for sentence in test_data:
            if len(sentence) > 32000:
                sentence = sentence[:32000]
            sentence_embedding = self.get_embedding(sentence)
            similarities = cosine_similarity([sentence_embedding], embeddings)
            if similarities[0][0] > similarities[0][1]:
                results.append(category1)
            else:
                results.append(category2)
        return results
