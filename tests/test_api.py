import unittest
from recommender.api import OpenWeatherMap, kelvin_to_celsius


class TestOpenWeatherMap(unittest.TestCase):

    def test_get_current_temp(self):
        owm = OpenWeatherMap()
        city = "Berlin"
        lat_long=[52, 12]
        try:
            temp = owm.get_current_weather(city)
            assert temp["temp"] > -20
            assert temp["temp"] < 40

            temp = owm.get_current_weather(lat_long=lat_long)
            assert temp["temp"] > -20
            assert temp["temp"] < 40

        except ConnectionError:
            assert False

    def test_api_weather_call(self):
        owm = OpenWeatherMap()
        resp = owm.api_weather_call("Berlin")
        assert resp is not None
        try:
            "temp" in resp["main"].keys()
        except KeyError:
            assert False

    def test_api_weather_call_lat_long(self):
        owm = OpenWeatherMap()
        resp = owm.api_weather_call_lat_long([52, 12])
        print(resp)
        assert resp is not None
        try:
            "temp" in resp["main"].keys()
        except KeyError:
            assert False

    def test_kelvin_to_celsius(self):
        assert kelvin_to_celsius(0) == -273.15
        assert kelvin_to_celsius(300) == 26.85
