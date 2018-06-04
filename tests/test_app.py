from recommender.gui.app import Recommend, Topic
import os
import unittest
import pandas as pd


class TestRecommend(unittest.TestCase):

    def setUp(self):
        self.r = Recommend()
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'destinations.csv')
        self.destinations_df = pd.read_csv(csv_path, sep=';', encoding="latin-1")

    def test_get_recommendation_LDA(self):
        res = self.r.get_recommendations_LDA(self.destinations_df, ["Berlin", "Florence", "Paris"])
        assert float(res[0][3]) + float(res[1][3]) + float(res[2][3]) >= 1, res

    def test_get_recommendation_LSI(self):
        res = self.r.get_recommendations_LSI(self.destinations_df, ["Berlin", "Florence", "Paris"])
        assert float(res[0][3]) + float(res[1][3]) + float(res[2][3]) >= 1, res

    def test_get_info(self):
        result_csv = [("Berlin", "bla1"), ("Paris", "blub"), ("Florence", "adsas")]
        dest_lat_long = {"Berlin": [52, 12], "Paris": [10, 10], "Florence": [43.7, 11.2]}
        res = self.r.get_info(result_csv, dest_lat_long)
        assert res[0][0] == "Berlin"
        assert len(res) == 3
        assert len(res[0]) == 5  # city, summary, weather, pic, top things to do
        assert res[0][2]["temp"] > -20
        print(res[2][2]["temp"])

    def test_get_summary(self):
        result_csv = [(1, 2, "BER", 0.5), (2, 3, "FRA", 0.4), (3, 4, "DBV", 0.3)]
        res = self.r.get_summary(self.destinations_df, result_csv)
        assert res[0][0] == "Berlin"
        assert len(res) == 3
        assert len(res[0]) == 2

    def test_get_lat_long(self):
        res = self.r.get_lat_long("Berlin")
        assert int(res[0]) == 52


class TestTopic(unittest.TestCase):
    def setUp(self):
        self.T = Topic()

    def test_topics(self):
        res = self.T.topics(["Berlin", "New York", "Barbados"])
        print(res)
        assert res[0][0] == "Berlin", res
        assert len(res) == 3
        assert len(res[0]) == 4
        assert any(["skyscrap" in x for x in res[1]])
        assert any(["island" in x for x in res[2]])


if __name__ == '__main__':
    unittest.main()
