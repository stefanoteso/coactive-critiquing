#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
from geopy.distance import vincenty
import xml.etree.ElementTree as ET
from itertools import product
from unidecode import unidecode

MAX_TRAVEL_TIME = 5
NUM_LOCATIONS = 30
# One time slot every 50 kilometers seems fair; it also means that a budget of
# 10 time slots equates to 500 kilometers, which also seems fair.
KM_THRESHOLD = 50
# The cost goes up by 1 every 100k tourists; threshold taken by manual
# inspection of a histogram
TOURIST_THRESHOLD = 100000

np.random.seed(0)


def sanitize(name):
    sanitized_name = unidecode(name.strip()).replace("'", "").upper()
    return sanitized_name


# Read the hotel data provided by:
#   http://dati.trentino.it/dataset/esercizi-alberghieri
ent_with_no_1_2_star_hotel = set()
ent_with_3_4_star_hotel = set()
ent_with_5_star_hotel = set()
root = ET.parse("datasets/tndata/EserciziAlberghieri.xml").getroot()
for child in root.findall("prezzi-localita-turistica"):
    ent = sanitize(child.attrib["denominazione"])
    for subchild in child.iter("prezzi-saa"):
        level = subchild.attrib["livello-classifica"]
        if level in ("nessuna-stella", "1-stella", "2-stelle"):
            ent_with_no_1_2_star_hotel.add(ent)
        elif level in ("3-stelle", "3-stelle-s", "4-stelle", "4-stelle-s"):
            ent_with_3_4_star_hotel.add(ent)
        elif level == "5-stelle":
            ent_with_5_star_hotel.add(ent)
        else:
            raise ValueError()

# Read the bus stop data provided by:
#   http://dati.trentino.it/dataset/trasporti-pubblici-del-trentino-formato-gtfs
df = pd.read_csv("datasets/tndata/bus_stops.csv", sep=",")
ent_with_bus_stop = set(map(sanitize,
                            [s.split("-")[0] for s in df.stop_name.unique()]))

# Read the agritur data provide by:
#   http://dati.trentino.it/dataset/agriturismi-del-trentino
df = pd.read_csv("datasets/tndata/elenco_agritur_attivi.csv", sep=";", engine="python")
ent_with_agritur = set(map(sanitize, df.sede_impresa_agricola.unique()))

# Read the museum data provided by:
#   https://it.wikipedia.org/wiki/Musei_del_Trentino-Alto_Adige
df = pd.read_csv("datasets/tndata/musei.csv", sep=",")
ent_with_museum = set(map(sanitize, df.ent.unique()))

# Read the osterie data provided by:
#   http://dati.trentino.it/dataset/osterie-tipiche-trentine
df = pd.read_csv("datasets/tndata/osterietipiche2013.csv", sep=";")
ent_with_osteria = set(map(sanitize,
                           [s.split("-")[0] for s in df.Comune.unique()]))

# Read the botteghe data provided by:
#   http://dati.trentino.it/dataset/botteghe-storiche-del-trentino
df = pd.read_csv("datasets/tndata/albobotteghestorichetrentinoluglio2016.csv", sep=",", engine="python")
ent_with_bottega = set(map(sanitize, df.Comune.unique()))

# Read the hospital data provided by:
#   http://dati.trentino.it/dataset/strutture-di-ricovero-e-cura
df = pd.read_csv("datasets/tndata/ospedali.csv", sep=",")
ent_with_ospedale = set(map(sanitize, df.COMUNE.unique()))

# Read the pharmacy data provided by:
#   http://dati.trentino.it/dataset/farmacie-pat
df = pd.read_csv("datasets/tndata/farmacie.csv", sep=",")
ent_with_farmacia = set(map(sanitize, df.COMUNE.unique()))

# Read the para-pharmacy data provided by:
#   http://dati.trentino.it/dataset/parafarmacie-pat
df = pd.read_csv("datasets/tndata/parafarmacie.csv", sep=",")
ent_with_parafarmacia = set(map(sanitize, df.COMUNE.unique()))

# Read the strutture sanitarie data provided by:
#   http://dati.trentino.it/dataset/strutture-sanitarie-dell-azienda-sanitaria-e-convenzionate
df = pd.read_csv("datasets/tndata/strutture_sanitarie.csv", sep=",")
ent_with_strutsan = set(map(sanitize, df.COMUNE.unique()))

# Read the strutture riabilitazione data provided by:
#   http://dati.trentino.it/dataset/strutture-di-riabilitazione
df = pd.read_csv("datasets/tndata/strutture_riabilitazione.csv", sep=",")
ent_with_strutriab = set(map(sanitize, df.COMUNE.unique()))

# Read the guardia medica data provided by:
#   http://dati.trentino.it/dataset/continuita-assitenziale-ex-gm
df = pd.read_csv("datasets/tndata/continuita_assistenziale.csv", sep=",")
ent_with_contassist = set(map(sanitize, df.COMUNE.unique()))

# Read the library data provided by:
#   http://dati.trentino.it/dataset/biblioteche-ai-censimenti
ent_with_library = set()
for line in open("datasets/tndata/biblioteche.xml", "rb"):
    line = line.decode().strip()
    if "<comune>" in line:
        ent = line.split(">")[1].split("<")[0]
        ent_with_library.add(sanitize(ent))



# Read the code->entity map provided by:
#   http://dati.trentino.it/dataset/elenco-codici-ente
df = pd.read_csv("datasets/tndata/codente50.csv", sep=";", comment="#")
ents = sorted(map(sanitize, df.descriz.unique()))
regs = sorted(map(sanitize, df.comvall.unique()))

ent_to_reg = {}
for ent, reg in zip(df.descriz, df.comvall):
    ent_to_reg[sanitize(ent)] = sanitize(reg)

ent_to_cost = {}
for ent, tou in zip(df.descriz, df.presenze):
    ent_to_cost[sanitize(ent)] = 1 + int(tou) // TOURIST_THRESHOLD

ent_to_i = {ent: i for i, ent in enumerate(ents)}
reg_to_i = {reg: i for i, reg in enumerate(regs)}

# Read the entity->(lat,lon) map
df = pd.read_csv("datasets/tndata/coords.csv", sep=",")

ent_to_coords = {}
for ent, lat, lon in zip(df.ent, df.lat, df.lon):
    ent_to_coords[sanitize(ent)] = (lat, lon)
assert all(ent in ent_to_coords for ent in ents)


# Build the location-activity array
ENT_WITH_ACTIVITIES = [
    ent_with_no_1_2_star_hotel,
    ent_with_3_4_star_hotel,
    ent_with_5_star_hotel,
    ent_with_agritur,
    ent_with_osteria,
    ent_with_bottega,
    ent_with_bus_stop,
    ent_with_library,
    ent_with_museum,
    ent_with_ospedale,
    ent_with_farmacia,
    ent_with_parafarmacia,
    ent_with_strutsan,
    ent_with_strutriab,
    ent_with_contassist,
]

num_locations = len(ents)
num_activities = len(ENT_WITH_ACTIVITIES)

location_activities = np.zeros((num_locations + 1, num_activities), dtype=int)
for i, ent in enumerate(ents):
    for j, ent_with_activity in enumerate(ENT_WITH_ACTIVITIES):
        if ent in ent_with_activity:
            location_activities[i, j] = 1

# Compute the location costs; we have no better way to do it
location_cost = np.zeros(num_locations + 1, dtype=int)
for i, ent in enumerate(ents):
    location_cost[i] = ent_to_cost[ent]

# Compute an approximate travel time based on geospatial distance in KM
travel_time = np.zeros((num_locations, num_locations), dtype=int)
for ent1, ent2 in product(ents, ents):
    i1, coords1 = ent_to_i[ent1], ent_to_coords[ent1]
    i2, coords2 = ent_to_i[ent2], ent_to_coords[ent2]
    km = vincenty(coords1, coords2).km
    travel_time[i1, i2] = 1 + km // KM_THRESHOLD
np.fill_diagonal(travel_time, 0)

# Compute the entity->region map
num_regions = len(regs)
regions = [reg_to_i[ent_to_reg[ent]] for ent in ents]

print("""\
location activities =
{location_activities}

location costs =
{location_cost}

travel time =
{travel_time}

regions =
{regions}
""".format(**locals()))

dataset = {
    "location_activities": location_activities,
    "location_cost": location_cost,
    "travel_time": travel_time,
    "num_regions": num_regions,
    "regions": regions,
}
with open("datasets/travel_tn.pickle", "wb") as fp:
    pickle.dump(dataset, fp)
