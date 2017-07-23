# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:41:41 2017

@author: Ran
"""
#read captial bike share station info from online xml
# http://feeds.capitalbikeshare.com/stations/stations.xml
def readStationXML():
    import pandas as pd
    import urllib
    document = 'http://feeds.capitalbikeshare.com/stations/stations.xml'
    file = urllib.request.urlopen(document)
    
    import xml.etree.ElementTree as ET
    tree = ET.parse(file)
    root = tree.getroot()

    station_address = []
    station_number = []
    station_lat = []
    station_long = []
    for station in root.findall('station'):
        station_address.append(station.find('name').text)
        station_number.append(station.find('terminalName').text)
        station_lat.append(station.find('lat').text)
        station_long.append(station.find('long').text)

    station_number = [int(i) for i in station_number]
    station_lat = [float(i) for i in station_lat]
    station_long = [float(i) for i in station_long]
    StationInfo = pd.DataFrame(data = {'station address': station_address,'station number': station_number,'lat': station_lat,'long': station_long})
    StationInfo = StationInfo[['station address','station number','lat','long']]
    return StationInfo
