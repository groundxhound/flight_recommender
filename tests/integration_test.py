import requests
import unittest


class TestIntegration(unittest.TestCase):

    def test_home(self):
        resp = requests.get("http://127.0.0.1")
        assert resp.status_code == 200
        # ... ?

    def test_recommend(self):
        resp = requests.post("http://127.0.0.1/recommend", data={"city": ["Berlin", "New York", "Barbados"]})
        assert resp.status_code == 200

    def test_index(self):
        resp = requests.get("http://127.0.0.1/index")
        assert resp.status_code == 200

    def test_topic(self):
        resp = requests.get("http://127.0.0.1/topic")
        assert resp.status_code == 200


if __name__ == "__main__":
    unittest.main()
