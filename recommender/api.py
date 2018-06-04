# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:18:20 2018

@author: jgolde
"""
###IMPORTS
import urllib3
import os.path
from .photoapi.unsplash.api import Api
from .photoapi.unsplash.auth import Auth
from .config import authorization_code
import requests
import re


###CLASS DESTINATION
###represents the API interface for requesting descriptions about destinations
class Destination(object):

    def __init__(self, iata_code, full_name):

        # set PoolManager from lib
        http = urllib3.PoolManager()
        # define path to respective source folder
        self.path_txtfiles_WIKI = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'destinations'),'wikipedia')

        # Reusable template which checks API for current iata_code (passed from app class, init method)
        # checks if file for this API already exists
        if not (os.path.exists(os.path.join(self.path_txtfiles_WIKI, "{0}.txt".format(iata_code)))):
            print('for ' + iata_code + '...')
            # API URL
            self.url = 'https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&titles={0}'.format(
                full_name)

            # try to get response from API and save it as .txt file in folder
            # else go on to next one
            try:
                response = http.request('GET', self.url)
                if response:
                    name_of_file = "{0}.txt".format(iata_code)
                    completeName = os.path.join(self.path_txtfiles_WIKI, name_of_file)

                    file = open(completeName, "w")
                    file.write(str(response.data))
                    file.close()
            except:
                pass

        else:
            pass


class OpenWeatherMap:

    def __init__(self):
        self.api_key = "68d7f84d6744846cc667072f75b19898"
        self.api_url = "http://api.openweathermap.org/data/2.5/weather?{query}&APPID=68d7f84d6744846cc667072f75b19898"
        self.cache = {}

    def get_current_weather(self, city=None, lat_long=None):
        """
        Does an api call
        :param city: string name of the city
        :param lat_long: length two first latitude second longitude
        :return: current temperature in Celsius
        """
        try:
            weather = self.cache[city]
        except KeyError:
            if lat_long:
                weather = self.api_weather_call_lat_long(lat_long)
            elif city:
                weather = self.api_weather_call(city)
                self.cache[city] = weather
            else:
                raise ValueError("Specify city or latitude longitude")

        try:
            temp = {'temp': kelvin_to_celsius(weather['main']['temp']),
                    'icon': weather['weather'][0]['icon'],
                    'description': weather['weather'][0]['description']}
        except KeyError:
            raise ValueError("Unknown city: " + city)
        return temp

    def api_weather_call_lat_long(self, lat_long):
        """
        Getting weather data from openweathermap api.
        :param  lat_long: length two first latitude second longitude
        :return: dict of the openweathermap response to current weather in city
        """
        query = "lat=" + str(round(lat_long[0],1)) + "&lon=" + str(round(lat_long[1],1))
        url = self.api_url.format(query=query)
        try:
            resp = requests.post(url).json()
        except requests.exceptions.ReadTimeout:
            raise ConnectionError
        return resp

    def api_weather_call(self, city):
        """
        Getting weather data from openweathermap api.
        :param city: string cityname
        :return: dict of the openweathermap response to current weather in city
        """
        url = self.api_url.format(query="q=" + str(city))
        try:
            resp = requests.post(url).json()
            self.cache[city] = resp
        except requests.exceptions.ReadTimeout:
            raise ConnectionError
        return resp


def kelvin_to_celsius(temp_kel):
    """
    Converts temperature from kelvin to celsius.
    :param temp_kel: temperature in kelvin
    :return: temperature in celsius
    """
    return round(temp_kel - 273.15, 2)


class PhotoAPI(object):

    def __init__(self):
        """
        Comment
        """
        self.client_id = "023a85f9329604da9abe9b0929938adbbcbb5258442c885173ee30e21df50530"
        self.client_secret = "910799e2c30c4f91819571992b857a024c2470044e52c01ce155c2cefaaa6750"
        self.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        self.code = authorization_code

        self.auth = Auth(self.client_id, self.client_secret, self.redirect_uri, code=self.code)
        self.api = Api(self.auth)

    def get_photo(self, city):
        photo_list = self.api.search.photos(city)
        photo_id = str(photo_list['results'][0])[10:-2]
        photo_link = "https://source.unsplash.com/{0}/1600x900".format(photo_id)

        return photo_link


class TopThingsToDo(object):

    def __init__(self):
        self.api_key = 'AIzaSyCfoIFnPU1jVjdXri4aWH1E-C64pEPtL8c'
        self.regex = r'"name" : "(.*?)",'
        print(self.api_key)

    def get_attractions(self, city):
        url = 'https://maps.googleapis.com/maps/api/place/textsearch/json?query=things+to+do+in+{0}&language=en&key={1}'.format(
            city, self.api_key)
        r = requests.get(url)
        topthingstodo = re.findall(self.regex, r.text)[:5]
        return topthingstodo
