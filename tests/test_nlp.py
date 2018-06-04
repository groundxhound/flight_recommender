import unittest
from recommender.nlp import LanguageProcessing, get_k_best_indices
from nltk.corpus import stopwords
import pandas as pd

# hack i found on so. bad style? Or rather the testing parameter in the init method?
class Empty(object):
    pass

#??? seems clunky
class fake_db_con():
    def __init__(self, data):
        self.graph = fgraph()
        self.destinations = data


class fgraph:
    def __init__(self):
        self.cypher = fcypher()


class fcypher:
    def begin(self):
        return ftx()


class ftx:
    def append(self, *args):
        pass

    def commit(self, *args):
        pass

    def process(self, *args):
        pass


class TestLanguageProcessing(unittest.TestCase):

    def test_prepare_data(self):
        #L_test = Empty()
        #L_test.__class__ = LanguageProcessing # this shouldn't be allowed
        L_test = LanguageProcessing.__new__(LanguageProcessing)
        L_test.documents = ["this word words a BLOCKCHAIN aren't"]
        L_test.stop_words = set(stopwords.words("English"))
        L_test.prepare_data(min_word_count=1, no_above_fraction=1)
        print(len(L_test.dictionary))
        assert set(list(L_test.dictionary.values())) == {"word", "blockchain"}
        #print(L_test.corpus)
        #print(L_test.tfidf)
        #print(L_test.corpus_tfidf)

    def test_number_of_topics_lsi(self):
        L = LanguageProcessing(testing=True)
        L.documents = ["bmw is a car", "vw is a car", "sun and beach", "a sun"]
        L.prepare_data(min_word_count=1, no_above_fraction=1)
        top = L.number_of_topics_lsi(tfidf=False)
        assert top == 2, top

    def test_create_topic(self):
        texts = ["bmw is a car", "vw is a car", "sun and beach", "a sun"]
        data = pd.DataFrame({"iata_code": ["a", "b", "c", "d"], "text": texts})
        test_db_con = fake_db_con(data)
        #clunky
        L = LanguageProcessing(test_db_con, config={"algorithm":"LSI", "min_word_count":1, "no_above_fraction":1, "nr_topics": 0,
                              "delete_numbers": False, "delete_words": False, "tfidf": False})

        topic = L.model.get_topics()[0, :]
        assert L.create_topic(topic, 0) == {"index": 0, "words_pos": ["car", "bmw", "vw"],
                                                "words_neg": ["sun", "beach", "vw"]}

    def test_doc_topic_representation(self):
        texts = ["bmw is a car", "vw is a car", "sun and beach", "a sun"]
        data = pd.DataFrame({"iata_code": ["a", "b", "c", "d"], "text": texts})
        test_db_con = fake_db_con(data)
        #clunky
        L = LanguageProcessing(test_db_con, config={"algorithm":"LSI", "min_word_count":1, "no_above_fraction":1, "nr_topics": 0,
                              "delete_numbers": False, "delete_words": False, "tfidf": False})

        doc_topic_representation = L.model[L.corpus]
        res = L.create_topic_doc_relation(doc_topic_representation[0], 0, nr_topics=1)
        assert "iata_code = 'a'" in res[0]
        assert "t.index = 0" in res[0]
        assert "r.weight = 1.22" in res[0]

    def test_save_topics(self):
        texts = ["bmw is a car", "vw is a car", "sun and beach", "a sun"]
        data = pd.DataFrame({"iata_code": ["a", "b", "c", "d"], "text": texts})
        test_db_con = fake_db_con(data)
        #clunky
        L = LanguageProcessing(test_db_con, config={"algorithm":"LSI", "min_word_count":1, "no_above_fraction":1, "nr_topics": 0,
                              "delete_numbers": False, "delete_words": False, "tfidf": False})

        L.save_topics()

    """def test_py2neo_queries(self):
        # i dont know how to actually make an integration test with neo4j
        from py2neo import Graph
        from recommender.config import neo4j_url
        graph = Graph(neo4j_url)
        tx = graph.cypher.begin()
        tx.append("Create (r:Topic { index: 1, word_neg: ['a', 'b'], word_pos: ['c', 'd']})")
        tx.process()
        tx.append("MATCH (d:Destination),(t:Topic) WHERE d.iata_code = 'NYC' AND t.index = 1 CREATE "
                  "(d)-[r:doc_topic]->(t) set r.weight = 2")
        tx.process()
        tx.commit()
        assert False""" #dont run it changes the database :S

    def test_get_k_best_indices(self):
        l = [2, 6, 1, 687, 3, 5]
        k = 3
        res = get_k_best_indices(l, k)
        assert (res==[3, 1, 5]).all(), res


if __name__ == "__main__":
    unittest.main()
