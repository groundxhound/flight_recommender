# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:34:04 2018

@author: jgolde
"""

###IMPORTS
import nltk
import os
import numpy as np
import pandas as pd
import gensim
from ast import literal_eval
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim import corpora, models, similarities
from recommender.config import config_def, score_model
import re


###CLASS LANGUAGE PROCESSING
###calculates the similarity between two descriptions
class LanguageProcessing:

    def __init__(self, db_connection=None, config=config_def, score_model=score_model, testing=False):

        # set necessary objects:
        # db_connection: technical connection to neo4j database, destinations: destinations data frame
        # documents: extracts the text from a unified column with all descriptions(not implemented yet)
        # stop_words: english stop words such as I with or and.
        self.stop_words = set(stopwords.words('english'))
        if not testing:
            self.db_connection = db_connection
            self.destinations = self.db_connection.destinations
            self.documents = list(self.destinations['text'])

            if score_model:
                self.model, self.res_corpus = None, None
                self.create_model(config=config)

    def create_model(self, config=config_def):
        """
        Creates an NLP model.
        :param config: dictionary with several parameters, can be changed in config.py (see there for doc)
        :return: modifies attributes of object
        """
        algorithm = config["algorithm"]
        nr_topics = config["nr_topics"]
        self.prepare_data(min_word_count=config["min_word_count"], no_above_fraction=config["no_above_fraction"],
                          delete_numbers=config["delete_numbers"], delete_words=config["delete_words"])

        if algorithm == "LSI":
            self.model, self.res_corpus = self.fit_lsi_model(nr_topics, tfidf=config["tfidf"])

        elif algorithm == "LDA":
            self.model, self.res_corpus = self.fit_lda_model(nr_topics, tfidf=config["tfidf"])

        else:
            raise ValueError("Specified Algorithm is not supported.")

        self.calculate_similarity(self.model)
        self.save_similarity_matrix(algorithm)

    def prepare_data(self, min_word_count=1, no_above_fraction=1, delete_numbers=False, delete_words=False):
        # Remove stop words and persons names and Get word stem
        # Extract persons' names
        self.names = []
        for doc in self.documents:
            self.names = self.extract_entities(self.names, doc)
        # Update the stop words list
        self.stop_words.update(self.names)
        if delete_words:
            self.stop_words.update(delete_words)
        # define word stemmer
        stemmer = SnowballStemmer("english")  # Choose a language

        # stem each word in documents and remove stop words
        if delete_numbers:
            self.texts = [[re.sub('[^a-zA-Z]+', '', stemmer.stem(word))
                           for word in document.lower().split()
                           if (word not in self.stop_words)] for document in self.documents]
        else:
            self.texts = [[stemmer.stem(word) for word in document.lower().split()
                           if (word not in self.stop_words)] for document in self.documents]

        self.texts = [[w for w in t if w != ""] for t in self.texts]
        # Create dictionary from stemmed words (self.texts)
        self.dictionary = corpora.Dictionary(self.texts)

        self.dictionary.filter_extremes(no_below=min_word_count, no_above=no_above_fraction)
        print(str(len(self.dictionary)) + " number of words used from texts.")
        # Create corpus (counts number of occurences of each distinct word
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        # fit the model
        self.tfidf = models.TfidfModel(self.corpus)
        # apply the model (vectorizing)
        self.corpus_tfidf = self.tfidf[self.corpus]

    def fit_lsi_model(self, nr_topics, tfidf):
        if not nr_topics:
            nr_topics = self.number_of_topics_lsi(tfidf)
            print(str(nr_topics) + " number of topics chosen for lsi.")
        # compute singular vectors by considering num of topics
        if tfidf:
            self.lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary,
                                       num_topics=nr_topics)
            return self.lsi, self.corpus_tfidf
        else:
            self.lsi = models.LsiModel(self.corpus, id2word=self.dictionary,
                                       num_topics=nr_topics)

            return self.lsi, self.corpus

    def number_of_topics_lsi(self, tfidf):
        """
        Takes the number of documents, number of words and the corpus to calculate a "good" number of topics
        with a minimum description length idea. Heuristic
        tfidf: boolean using tf-idf corpus or regular
        :return: int number of topics to use in the lsi algorithm
        """
        nr_texts = len(self.documents)
        nr_words = len(self.dictionary)
        if tfidf:
            corpus = self.corpus_tfidf
        else:
            corpus = self.corpus

        words_text_mat = gensim.matutils.corpus2dense(corpus, nr_words)

        # singular value decomposition
        s = np.linalg.svd(words_text_mat, full_matrices=False, compute_uv=False)
        # sorting them
        s = -1 * np.sort(-1*s)
        # The squared sum of some of the singular values (eigenvalues) divided by the total sum is the variance
        # explained by those components
        s2 = s ** 2
        cum_s2 = np.cumsum(s2) / sum(s2)
        model_quality = -1 * np.array(range(1, nr_texts + 1)) / nr_texts + cum_s2
        num_topics = np.argmax(model_quality)
        return num_topics + 1  # counting from zero

    def fit_lda_model(self, nr_topics, tfidf):

        self.Lda = gensim.models.ldamodel.LdaModel
        if tfidf:
            corpus = self.corpus_tfidf
        else:
            corpus = self.corpus
        self.ldamodel = self.Lda(corpus, num_topics=nr_topics, id2word=self.dictionary, passes=50)

        return self.ldamodel, corpus

    def save_similarity_matrix(self, algorithm):

        # save calculation to csv file
        self.path_dest_sim = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.similarity2csv = self.destinations[['iata_code', 'similarity']]
        self.similarity2csv.to_csv(os.path.join(self.path_dest_sim, 'destination_similarity.csv'), sep='|')

        # read in the similarity data (needed writing it into neo4j)
        self.similarity_df = pd.read_csv(os.path.join(self.path_dest_sim, 'destination_similarity.csv'), sep='|')

        # write into neo4j
        tx = self.db_connection.graph.cypher.begin()
        if algorithm == "LSI":

            statement = ("MATCH (d_ref:Destination {iata_code:{A}}) "
                         "MATCH(d:Destination {iata_code:{B}}) MERGE (d_ref)-[r:`Similarity_LSI`]-(d) Set r.score={C}")

        elif algorithm == "LDA":

            statement = ("MATCH (d_ref:Destination {iata_code:{A}}) "
                         "MATCH(d:Destination {iata_code:{B}}) MERGE (d_ref)-[r:`Similarity_LDA`]-(d) Set r.score={C}")

        # Looping over destinations
        for index, d in self.similarity_df.iterrows():
            # Transform the m.similarity into a list
            similarity = literal_eval(d.similarity)
            # Looping over destinations similar to d
            for sim in similarity:
                if sim[0] != d['iata_code']:
                    tx.append(statement, {"A": d['iata_code'], "B": sim[0], "C": sim[1]})
            tx.process()
            if index % 10 == 0:
                tx.commit()
                tx = self.db_connection.graph.cypher.begin()
        tx.commit()

    def calculate_similarity(self, model):

        self.model = model
        self.index = similarities.MatrixSimilarity(self.model[self.corpus])
        self.destinations['similarity'] = 'unknown'
        # storage of all similarity vectors to analysis
        self.total_sims = []
        threshold = 0.2
        # loop over corpus, for each vector
        for i, doc in enumerate(self.corpus):
            # convert the vector to LDA space
            vec_model = self.model[doc]
            # perform a similarity vector against the corpus
            sims = self.index[vec_model]
            self.total_sims = np.concatenate([self.total_sims, sims])
            # Create a list with destination_id and similarity value
            similarity = []
            # determine for each connection between all destinations similarity between descriptions and save
            for each, iata_code in enumerate(self.destinations['iata_code']):
                if sims[each] > threshold:
                    similarity.append((iata_code, sims[each]))
            similarity = sorted(similarity, key=lambda item: -item[1])
            self.destinations.at[i, 'similarity'] = similarity

    def extract_entities(self, name, text):
        # Lopping over the sentences of the text
        for sent in nltk.sent_tokenize(text):
            # nltk.word_tokeize returns a list of words composing a sentence
            # nltk.pos_tag returns the position tag of words in the sentence
            # nltk.ne_chunk returns a label to each word based on this position tag when possible
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                try:
                    if chunk.label() == 'PERSON':
                        for c in chunk.leaves():
                            if str(c[0].lower()) not in name:
                                name.append(str(c[0]).lower())
                except AttributeError:
                    pass
        return name

    def optimize_parameters(self):
        """
        Parameter tuning grid search. Specify what input you would expect to be similar (test_expected) for some
        example input (test_input). Running all combinations of parameter choices that are specified.
        :return: The configuration that rated the expected values the highest. (overfitting)
        """
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'destinations.csv')
        destinations_df = pd.read_csv(csv_path, sep=';', encoding="latin-1")
        test_input = [["Barbados", "Mauritius", "Palma de Mallorca"], ["New York", "Los Angeles", "Berlin"]]
        # ["Phuket", "Malta", "Ibiza], ["Paris", "London", "San Francisco"]
        test_expected = [["HKT", "MLA", "IBZ"], ["PAR", "LON", "SFO"]]
        best_config = {}
        best_score = 0
        for nr_topics in [5, 15, 30, 60]:
            for algorithm in ["LSI", "LDA"]:
                for tfidf in [True, False]:
                    for min_word_count in [6]:
                        for no_above_fraction in [0.4]:
                            for delete_numbers in [True]:
                                for delete_words in [["also"]]:
                                    config = dict(zip(["algorithm", "min_word_count", "no_above_fraction",
                                                       "delete_numbers", "delete_words", "nr_topics", "tfidf"],
                                                      [algorithm, min_word_count, no_above_fraction, delete_numbers,
                                                       delete_words, nr_topics, tfidf]))

                                    score = self.score_config(config, test_input, test_expected, destinations_df)
                                    print(config)
                                    print(score)
                                    if score > best_score:
                                        best_config = config
                                        best_score = score
        print("best config is: " + str(best_config))
        return best_config

    def score_config(self, config, test_input, test_expected, destinations):
        """
        Counts how many of the expected values where in the top 15 recommendations.
        :param config: config to be used for the model
        :param test_input: names that specify documents
        :param test_expected: Documents that should be similar to the input
        :param destinations: pd.DataFrame to match iata_codes and city names
        :return: score of the configuration
        """
        tx = self.db_connection.graph.cypher.begin()
        tx.append("Match (a:Destination)-[r]-(b:Destination) detach delete r")
        tx.commit()
        self.create_model(config)
        from recommender.gui.app import Recommend
        r = Recommend()
        score1 = 0.
        score2 = 0
        algorithm = config["algorithm"]
        for i, choices in enumerate(test_input):
            recomm = r.get_recommendations(destinations, choices, algorithm)
            ranks = [x[0] for x in list(recomm) if x[2] in test_expected[i]]
            score1 += sum([-1 * float(x) for x in ranks])
            score2 += sum([1 for x in ranks if int(x) <= 15])
        return score2

    def save_topics(self, nr_words=3, nr_topics=3):
        """
        Create topics and relations from topics to documents in neo4j.
        :param nr_words: Number of words to show for each topic
        :param nr_topics: Number of topics to create relationships for each document
        :return: None
        """
        delete_old_topics = "match (t:Topic) detach delete t"
        topics = self.model.get_topics()
        tx = self.db_connection.graph.cypher.begin()
        tx.append(delete_old_topics)
        for topic_index in range(topics.shape[0]):
            attr_dict = self.create_topic(topics[topic_index, :], topic_index, nr_words)
            # need to escape the curly braces by doubling it ... dis bug
            create_statement = "create (n:Topic {{ index: {index}, word_pos: {words_pos}, word_neg:{words_neg}}})"
            statement = create_statement.format(**attr_dict)
            tx.append(statement)
            if topic_index % 10 == 0:
                tx.process()
        tx.commit()
        tx = self.db_connection.graph.cypher.begin()
        doc_topic_representation = self.model[self.corpus]
        for doc_index in range(len(doc_topic_representation)):
            queries = self.create_topic_doc_relation(doc_topic_representation[doc_index], doc_index, nr_topics)
            for s in queries:
                tx.append(s)
            if doc_index % 10 == 0:
                tx.process()
        tx.commit()

    def create_topic(self, topic, topic_index, nr_words=3):
        """
        Retrieves the most important words of a topic.
        :param topic: word vector of a topic
        :param topic_index: unique index of the topic, kind of the name of the topic
        :param nr_words: extract that many important words of the topic
        :return: dict with the index, words_pos the nr_words most important if topic has positiv weight
                words_neg the nr_words most important words if topic has negative weight
        """
        attr_dict = {"index": topic_index}
        topic = np.round(topic, 2)
        x = [(self.dictionary[j], topic[j]) for j in range(len(topic))]  # if abs(v1[i]) > 0.1]
        x.sort(key=lambda a: a[1], reverse=True)

        attr_dict["words_pos"] = [y[0] for y in x[0:nr_words]]

        tmp = x[-nr_words:]
        tmp.reverse()
        attr_dict["words_neg"] = [y[0] for y in tmp]

        return attr_dict

    def create_topic_doc_relation(self, doc_topic_vec, doc_index, nr_topics=3):
        """
        Finds the most important topics for the input document.
        :param doc_topic_vec: vector with weights for each topic.
        :param doc_index: document index
        :param nr_topics: Number of topics for each document to create relationships with.
        :return: query string
        """
        iata_code = self.destinations["iata_code"].iloc[doc_index]
        topic_vec = [x[1] for x in doc_topic_vec]
        topic_vec = np.round(topic_vec, 2)
        topic_vec_abs = np.abs(topic_vec)
        nr_topics = min(len(topic_vec), nr_topics)
        topic_indices = get_k_best_indices(topic_vec_abs, nr_topics)
        create_rel = []
        for i in topic_indices:
            create_rel.append("MATCH (d:Destination),(t:Topic) WHERE d.iata_code = '{iata_code}' AND "
                              "t.index = {topic_index} CREATE (d)-[r:doc_topic]->(t) set r.weight = {weight}".format(
                iata_code=iata_code, topic_index=i, weight=topic_vec[i]))
        return create_rel


def get_k_best_indices(l, k):
    """
    Finds the indices of the k biggest elements of the list l
    :param l: list/np.array of numbers
    :param k: integer, the best k indices returned
    :return: np.array of indices
    """
    # sorts the whole array nlogn :S it works, don't judge pls
    return np.argsort(l)[-k:][::-1]
