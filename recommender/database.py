# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:25:18 2018

@author: jgolde
"""

###IMPORTS
import re
import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from py2neo import Graph, cypher
from os.path import join, isfile
from os import listdir

###CLASS DATABASE
###save collected data (txt files from API) in neo4j graphical database
class Database():
    
    def __init__(self):
        
        #connect to running neo4j server at port 7474
        self.graph = Graph("http://neo4j:neo4j@localhost:7474/db/data/")
        
        #set paths for all destinations per API e.g. PATH_WIKI = ...//destinations//wikipedia
        #destination files set up manually
        recommender_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path_txtfiles = os.path.join(recommender_path, 'destinations')
        #all destinations which are written into database - DIFFERENT CSV FILE THAN FOR API
        #currently done out of quality reasons. Aim ist to use similar file.


        self.path_all_dest = os.path.join(recommender_path, 'destinations.csv')
        #Create dataframe with columns: iata_code, full_name and area (continental)
        self.destinations = pd.read_csv(self.path_all_dest, encoding='latin-1', sep=';')
        
        #create here dataframe for each data request per API source, e.g. ..(columns=[VAR NAME])
        #naming: description_API SOURCE
        self.descriptions = pd.DataFrame(columns=['text'])
        
        #loop over all destinations from csv of which we know we have data with quality
        #index: 1-x, dest: dict{iata_code, full_name, area}
        for index, dest in self.destinations.iterrows():
            #checks for first API source if file is found in destinations data folder
            #dest[0] = iata_code
            path_current_file = os.path.join(self.path_txtfiles, str('.'.join([dest[0],'txt'])))
            #if exists: open file and read as tmp var, finally append to data frame as new column
            #else pass to next one
            if os.path.exists(path_current_file):
                with open(path_current_file) as tmp_f:
                    try:
                        description = tmp_f.read()
                    except:
                        pass
                self.descriptions.loc[index] = [description]
            else:
                self.descriptions.loc[index] = [""]
        else:
            pass
        
        #join dataframe destinatination with all descriptions from api
        self.destinations = pd.concat([self.destinations, self.descriptions], axis=1)
        
        #tokenizer as Regular Expression tokenizer
        tokenizer = RegexpTokenizer(r'\w+')
        #replace all description columns with it tokenized selfs
        self.destinations['text'] = self.destinations['text'].apply(lambda x: ' '.join(tokenizer.tokenize(x)))
        
        #write destination with their tokenized descriptions into neo4j
        tx = self.graph.cypher.begin()
        statement = "MERGE (a:Destination{destination_id:{A}, iata_code:{B}, description:{C}, area:{D}}) RETURN a"
        for d, row in self.destinations.iterrows():
            tx.append(statement, {"A": d, "B": str(row.loc['iata_code']), "C": str(row.loc['text']), "D": str(row.loc['area'])})
        tx.commit()
        
        
        
        #OLD CODE
        #LOADING DESTINATIONS
        #self.path_txtfiles_LH = str(os.getcwd()+"\\destinations\\")
        #self.path_txtfiles_WIKI = str(os.getcwd()+"\\destinations\\wikipedia\\")
        #self.path_all_dest = str(os.getcwd()+"\\dest\\destinations_new.csv")
        
        #self.destinations = pd.read_csv(self.path_all_dest, encoding='latin-1', sep=';')
        
        #GET ALL THE TEXTS FROM LUFTHANSA
        #self.descriptions_LH = pd.DataFrame(columns=['text_LH'])

        #for index, dest in self.destinations.iterrows():
        #   path_current_file = self.path_txtfiles_LH + str('.'.join([dest[0],'txt']))
        #    if os.path.exists(path_current_file):
        #        with open(path_current_file) as tmp_f:
        #            try:
        #                description = tmp_f.read()
        #            except:
        #                pass
        #            self.descriptions_LH.loc[index] = [description]
        #    else:
        #        self.descriptions_LH.loc[index] = [""]
                
        #self.destinations = self.destinations.join(self.descriptions_LH)
                
        ##GET ALL THE TEXTS FROM WIKIPEDIA
        #self.descriptions_WIKI = pd.DataFrame(columns=['text_WIKI'])
        #for index, dest in self.destinations.iterrows():
        #    path_current_file = self.path_txtfiles_WIKI + str('.'.join([dest[0],'txt']))
        #    if os.path.exists(path_current_file):
        #        with open(path_current_file) as tmp_f:
        #            try:
        #                description = tmp_f.read()
        #            except:
        #                pass
        #            self.descriptions_WIKI.loc[index] = [description]
        #    else:
        #        self.descriptions_WIKI.loc[index] = [""]
                
        #self.destinations = self.destinations.join(self.descriptions_WIKI)
        #self.destinations['text'] = self.destinations['text_WIKI'] + self.destinations['text_WIKI']
    
        #tokenizer = RegexpTokenizer(r'\w+')
        #self.destinations['text'] = self.destinations['text'].apply(lambda x: ' '.join(tokenizer.tokenize(x)))
        #self.destinations['text_LH'] = self.destinations['text_LH'].apply(lambda x: ' '.join(tokenizer.tokenize(x)))
        #self.destinations['text_WIKI'] = self.destinations['text_WIKI'].apply(lambda x: ' '.join(tokenizer.tokenize(x)))

        #tx = self.graph.cypher.begin()
        #statement = "MERGE (a:Destination{destination_id:{A}, iata_code:{B}, description:{C}}) RETURN a"
        #for d, row in self.destinations.iterrows():
        #    tx.append(statement, {"A": d, "B": str(row.loc['iata_code']), "C": str(row.loc['text_LH'])})
        #tx.commit()
        
        #    def recommend_destination(self, selected_destinations, threshold):
            
        #        query = (#Identify destinations visited
        #                     "MATCH (visited:Destination) WHERE visited.destination_id = {dest1} "
        #                     "AND visited.destination_id = {dest2} AND visited.destination_id = {dest3} "
        #                     "WITH collect(visited.destination_id) as visited_set "
        #                     #Identify destinations not visited
        #                     "MATCH (unvisited:Destination) WHERE NOT unvisited.destination_id = 3 "
        #                     "AND NOT unvisited.destination_id = 5 AND NOT unvisited.destination_id = 1 "
        #                     "WITH collect(unvisited.destination_id) as unvisited_set "
        #                     #Calc recommendations
        #                     "MATCH (visited)-[sim:Similarity]-(reco:Destination) "
        #                     "WHERE reco.destination_id in unvisited_set and sim.score > (0.6) "
        #                     "RETURN DISTINCT reco.destination_id as destination_id, reco.iata_code, sum(sim.score) as score "
        #                     "ORDER BY score DESC")
        #        tx = self.graph.cypher.begin()
        #        tx.append(query, {"dest1":, "dest2": , "dest3":, "threshold":int(threshold)})
        #        result = tx.commit()