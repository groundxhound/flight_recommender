###IMPORTS
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from flask import Flask, render_template, request
from flask.views import View
from py2neo import Graph
import requests
import operator
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map
import pandas as pd
import os
import re
from ..api import OpenWeatherMap, PhotoAPI, TopThingsToDo
from ..database import Database
from ..nlp import LanguageProcessing
from recommender.config import algorithm
from recommender.config import create_topics, photoapi_disabled


###CLASS APP:
###intializes entire program including front end running on specified port
class App():
    # all destinations for frontend selection
    destinations_dict = {}
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'destinations.csv')
    destinations_df = pd.read_csv(csv_path, sep=';', encoding="latin-1")
    for i in destinations_df.index.values:
        destinations_dict[destinations_df['full_name'][i]] = i

    # OLD DESTINATIONS
    # destinations = {"Rome": 1, "Porto": 6, "Paris": 4, "Berlin": 5,
    #                "Florence": 8,
    #                "Prague": 7, "Barcelona": 10, "Dubai": 9, "Santorini": 14,
    #                "Vancouver": 11, "Warsaw": 15,
    #                "Mauritius": 17, "Shanghai": 19, "Seoul": 22,
    #                "Brussels": 21, "Munich": 25, "Toronto": 24,
    #                "Genoa": 27, "Amsterdam": 26, "Oslo": 29, }
    #                # no information in destination file
    #                # "Sydney": 3, "Glasgow": 2, "Honolulu": 13, "Bali": 12, "Salvador": 18,
    #                # "Belize": 23, "Ljubljana": 30, "Pittsburgh": 28, "Skopje": 20}

    # variable which is passed in any return object when requesting a recommendation
    dest_ranked = sorted_x = sorted(destinations_dict.items(), key=operator.itemgetter(1))

    # determine object specific attributes
    def __init__(self):
        # Flask initialisation
        self.app = Flask(__name__)
        # Google maps API key (necessary for requesting map data)
        self.app.config['GOOGLEMAPS_KEY'] = "AIzaSyBkFBjMxRZsKjoqDC9DrH0VWn2PNLxLdJo"

        # Initialize the extension
        GoogleMaps(self.app)

        # add links on your flask app here in the format:
        # self.app.add_url_rule('URL NAME', view_func=CLASS NAME OF EXTENSION.as_view('REPRESENTATIVE CLASS NAME'))
        self.app.add_url_rule('/', view_func=Intro.as_view(''))
        self.app.add_url_rule('/index', view_func=Home.as_view('index'))
        self.app.add_url_rule('/bot', view_func=Recommend.as_view('bot'))
        self.app.add_url_rule('/recommend', view_func=Recommend.as_view('recommend'))
        self.app.add_url_rule('/topic', view_func=Topic.as_view('topic'))
        # reads all destinations from all_destinations.csv for wikipedia api
        # tmp_df = pd.read_csv(str(os.getcwd() + '//all_destinations.csv'), sep=';', encoding="latin-1")
        # tmp_df.replace('\s+', '_',regex=True,inplace=True)
        # self.dest_as_list = [tmp_df.loc[x, :].values.tolist() for x in range(len(tmp_df)) if not pd.isnull(tmp_df.loc[x][1])]
        # tmp_df = None

        # loop over previous defined list and extract all destinations information via REST API if not exists already
        # AIM: centralize all APIs in scirpt destinations_api.py to extract information and safe as .txt
        # print('setting up information about total {0} destinations'.format(str(len(self.dest_as_list))))
        # self.destinations_wikipedia = {}
        # for each_destination in self.dest_as_list:
        #    self.destinations_wikipedia[each_destination[0]] = Destination(each_destination[0], each_destination[1])   

        # Write collected data into database (neo4j)
        print('setting up database...')
        self.db_connection = Database()

        # Calculate similarity between destination texts
        print('calculation similarity...')
        self.language_module = LanguageProcessing(self.db_connection)
        if create_topics:
            self.language_module.save_topics(3, 10)

        # Setting up flask app on specified port
        print('setting up user interface...')
        self.app.run(host='0.0.0.0', port=80, debug=False)


###FLASK APP CLASS
###view Home is initial / Homepage		
class Intro(View):

    def dispatch_request(self):
        # return index view with dest_ranked as parameter
        return render_template('intro.html')


class Home(View):

    def dispatch_request(self):
        # return index view with dest_ranked as parameter
        return render_template('index.html', data=App.dest_ranked, flag=0)


class Topic(View):
    methods = ['GET', 'POST']

    def __init__(self):
        from recommender.config import neo4j_url
        self.graph = Graph(neo4j_url)

    def dispatch_request(self):
        if request.method == "GET":
            return self.get()
        elif request.method == "POST":
            choices = request.form.getlist('city')
            topics = self.topics(choices)
            return render_template('topic.html', data=topics, flag=1)

    def get(self):
        return render_template('topic.html', data=App.dest_ranked, flag=0)

    def topics(self, choices, nr_topics=3):
        tx = self.graph.cypher.begin()
        name_iata_code_dic = dict(zip(App.destinations_df["full_name"], App.destinations_df["iata_code"]))
        topic_list = []
        for city_name in choices:
            iata_code = name_iata_code_dic[city_name]
            print(iata_code)
            statement = "Match (d:Destination)-[r:doc_topic]-(t:Topic) where d.iata_code = '{iata_code}' return" \
                        " t.word_pos, t.word_neg, r.weight order by abs(r.weight) Desc limit {nr_topics}"
            tx.append(statement.format(iata_code=iata_code, nr_topics=nr_topics))
            res = tx.process()
            one_city_topics = [city_name]
            for row in res:
                for x in row:
                    if x["r.weight"] > 0:
                        one_city_topics.append(', '.join(x["t.word_pos"]))
                    else:
                        one_city_topics.append(', '.join(x["t.word_neg"]))
            topic_list.append(one_city_topics)
        return topic_list


###FLASK APP CLASS
###view Recommend is what happens when you click on recommend
class Recommend(View):
    # define html events and set database connection to neo4j
    from recommender.config import neo4j_url
    methods = ['GET', 'POST']
    graph = Graph(neo4j_url)

    def dispatch_request(self):

        # check whether action event is post
        if request.method == 'POST':
            # get selected data from webapp as list (e.g. (Rome, Lisbon, Berlin))            
            date = request.form.get('startdate')
            duration = request.form.get('duration')
            choices = request.form.getlist('city')
            continents = request.form.getlist('continent')
            activity_style = request.form.get('activities_style')
            
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    'destinations.csv')

            destinations = pd.read_csv(csv_path, sep=';', encoding="latin-1")
            result_csv = self.get_recommendations(destinations, choices, algorithm)

            # transform recommendations (return in iata_code) to full name of cities
            choice_data = self.get_summary(destinations, result_csv)

            # request map locations from google
            map_data, dest_lat_long = self.mapview(choice_data)
            city_data = self.get_info(choice_data, dest_lat_long)

            # place calculated and requested data on view
            return render_template('index.html', map=map_data, choices=city_data, flag=1)

    def get_recommendations(self, destinations, choices, algorithm):

        if algorithm == "LSI":
            return self.get_recommendations_LSI(destinations, choices)
        elif algorithm == "LDA":
            return self.get_recommendations_LDA(destinations, choices)
        else:
            raise ValueError("Specified Algorithm is not supported.")

    def get_recommendations_LSI(self, destinations, choices):
        # save the looked up cities in choice_data_iata
        # this transformation is necessary for right query design of neo4j
        choice_data_iata = []
        for full_name in choices:
            choice_data_iata.append(destinations.loc[destinations['full_name'] == full_name, 'iata_code'].iloc[0])

        query = (  # Identify destinations visited
            "MATCH (selected:Destination) "
            "WHERE selected.iata_code IN [{dest1},{dest2},{dest3}] "
            # Sum up similarity of destinations returned
            "MATCH p=(selected)-[sim:Similarity_LSI]-(reco:Destination) "
            # at some point: create selection dynamic
            "WHERE NOT reco.iata_code IN [{dest1}, {dest2}, {dest3}] "
            "RETURN reco.destination_id, reco.iata_code, sum(sim.score) "
            "ORDER BY sum(sim.score) DESC"
        )

        # place neo4j query to request top 3 recommendations
        tx = self.graph.cypher.begin()
        tx.append(query, {"dest1": str(choice_data_iata[0]),
                          "dest2": str(choice_data_iata[1]),
                          "dest3": str(choice_data_iata[2])
                          })
        result = tx.commit()

        # parse the output into a readable dataframe
        return self.parse_output(result)

    def get_recommendations_LDA(self, destinations, choices):
        """
        Using the LDA model to recommend similar cities.
        :param choices: list of 3 city_names
        :return: list of 3 receommended city_names
        """
        # same as LSI but will work on changing it
        choice_data_iata = []
        for full_name in choices:
            choice_data_iata.append(destinations.loc[destinations['full_name'] == full_name, 'iata_code'].iloc[0])

        query = (  # Identify destinations visited
            "MATCH (selected:Destination) "
            "WHERE selected.iata_code IN [{dest1},{dest2},{dest3}] "
            # Sum up similarity of destinations returned
            "MATCH p=(selected)-[sim:Similarity_LDA]-(reco:Destination) "
            # at some point: create selection dynamic
            "WHERE NOT reco.iata_code IN [{dest1}, {dest2}, {dest3}] "
            "RETURN reco.destination_id, reco.iata_code, sum(sim.score) "
            "ORDER BY sum(sim.score) DESC"
        )

        # place neo4j query to request top 3 recommendations
        tx = self.graph.cypher.begin()
        tx.append(query, {"dest1": str(choice_data_iata[0]),
                          "dest2": str(choice_data_iata[1]),
                          "dest3": str(choice_data_iata[2])
                          })
        result = tx.commit()

        # parse the output into a readable dataframe
        return self.parse_output(result)

    def get_summary(self, destinations, result_csv):
        choice_data = []

        for ind, dest_id, iata, score in result_csv[:3]:
            dest_name = destinations.loc[destinations['iata_code'] == iata, 'full_name'].iloc[0]

            try:
                destination_description = open(os.getcwd() + '//destinations//{0}.txt'.format(iata)).read()
                split_up_sentences = re.findall("[A-Z].*?[\.!?]", destination_description, re.MULTILINE | re.DOTALL)
                summary = split_up_sentences[0] + split_up_sentences[1]
            except:
                summary = "Summary not found."
            choice_data.append((dest_name, summary))
            print(choice_data)
        return choice_data

    def get_info(self, choice_data, dest_lat_long):
        """
        Retrieve weather, photo and things to do data from apis.
        :param choice_data: list of lists containing the city name first and summary second
         [["Berlin", "Berlin is a city"], ...]
        :param dest_lat_long: dictionary with keys city names and values list of length city lat long values
        :return: list of lists containing city name, summary, temperature, photo_link and top things to do for each city
        """
        owm = OpenWeatherMap()
        tttd = TopThingsToDo()
        if not photoapi_disabled:
            pht = PhotoAPI()

        city_data = []
        for choice in choice_data:
            dest_name = choice[0]
            summary = choice[1]
            lat_long = dest_lat_long[dest_name]
            try:
                temp = owm.get_current_weather(lat_long=lat_long)
            except:
                temp = "Temperature not found."

            try:
                topthingstodo = tttd.get_attractions(dest_name)
            except:
                topthingstodo = "Attractions not found."

            if not photoapi_disabled:
                try:
                    photo_link = pht.get_photo(dest_name)
                except:
                    photo_link = photo_link = "../static/{0}.jpg".format(dest_name)
            else:
                pht_path = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gui'), 'static'),'{0}.jpg'.format(dest_name))
                if os.path.isfile(pht_path):
                    photo_link = "../static/{0}.jpg".format(dest_name)
                else:
                    photo_link = "../static/td5.jpg"

            city_data.append((dest_name, summary, temp, photo_link, topthingstodo))
        return city_data

    # requests the lat / long of a destination via google maps api
    def get_lat_long(self, city_name):
        """
        Retrieves latitude and longitude values from googleapis for a city name.
        :param city_name: string, name of a city
        :return: list length two with lat and long value
        """
        address = city_name
        api_key = "AIzaSyBkFBjMxRZsKjoqDC9DrH0VWn2PNLxLdJo"
        api_response = requests.get(
            'https://maps.googleapis.com/maps/api/geocode/json?address={0}&key={1}'.format(address, api_key))
        api_response_dict = api_response.json()

        if api_response_dict['status'] == 'OK':
            latitude = api_response_dict['results'][0]['geometry']['location']['lat']
            longitude = api_response_dict['results'][0]['geometry']['location']['lng']
            return [latitude, longitude]

    # creates the corresponding flask view on our app interface
    def mapview(self, city_info):
        """
        Creates googlemaps view for city_info
        :param city_info: generator/list of list with first two entries city name and what should be displayed as text
                            about that city.
        :return: Map object, dictionary with keys city names and values lat long list of two values
        """
        data = [('http://maps.google.com/mapfiles/ms/icons/green-dot.png', i[0], i[1]) for i in city_info]
        marker = []
        dest_lat_long = {}
        for city in data:
            lat_long = self.get_lat_long(city[1])
            marker.append({
                'icon': city[0],
                'lat': lat_long[0],
                'lng': lat_long[1],
                'infobox': "<b>" + city[1] + "</b>" + "<p>" + city[2] + "</p>"
            })
            dest_lat_long[city[1]] = lat_long
        maps = Map(
            identifier="travel_map",
            lat=40.730610,
            lng=-73.935242,
            markers=marker,
            style="height:500px; width:auto;",
            fit_markers_to_bounds=True
        )
        return maps, dest_lat_long

    # parses the output of recommendations into something readable
    def parse_output(self, reco_result):
        # RegEx definition
        pattern_score = r"(\d+.\d+)"
        pattern_index = r"(\d+)"
        pattern_iata = r"\w+"

        # PREP
        # set initial lists and delete header lines
        recommendations_csv = []
        to_csv = str(reco_result).split("\n")[2:]

        # split lines that we have a list index for each destination
        for each_line in to_csv:
            recommendations_csv.append(each_line.split("|"))

        # Creating actual list
        output_csv = []
        for item in recommendations_csv:
            # get index
            try:
                gotdata = item[0]
                row = re.findall(pattern_index, gotdata)
            except IndexError:
                gotdata = 'null'

            # get destination_id
            try:
                gotdata = item[1]
                row = row + re.findall(pattern_index, gotdata)
            except IndexError:
                gotdata = 'null'

            # get iata code
            try:
                gotdata = item[2]
                row = row + re.findall(pattern_iata, gotdata)
            except IndexError:
                gotdata = 'null'

            # get score
            try:
                gotdata = item[3]
                row = row + re.findall(pattern_score, gotdata)
            except IndexError:
                gotdata = 'null'
            if row:
                output_csv.append(row)

        return output_csv

    ###CLASS BOT


###for future development: chatbot interface should be placed here
class bot(View):

    def dispatch_request(self):
        return render_template('chatbot.html')
