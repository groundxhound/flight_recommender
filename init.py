#!/usr/bin/python

""" Start user interface for live candle plot. 
    @author: Jonas Golde
    @date: 01-02-2018
"""


#Import folder recommender - Must be on same logical level as starter file
from recommender import Gui

# trade and candle data will be saved to a data directory
gui = Gui()

# optional; initialize GUI with API key
#gui.setApiKey('your API "key" ', 'your API "secret" ')