#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# XXX this script is not exactly great; most checks and conversions are done
# in a best effort fashion

import pickle
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from unidecode import unidecode

MAX_TRAVEL_TIME = 5


def sanitize(name):
    return unidecode(name.strip()).replace("'", "").upper()


# Read the code->entity map provided by:
#   http://dati.trentino.it/dataset/elenco-codici-ente
df = pd.read_csv("tndata/codente.csv", sep=";")

cod_to_ent = {}
for cod, ent in zip(df.comu, df.descriz):
    if 1 <= cod < 999:
        cod_to_ent[cod] = sanitize(ent)

cods = set(cod_to_ent.keys())
ents = set(cod_to_ent.values())



# Read the hotel data provided by:
#   http://dati.trentino.it/dataset/esercizi-alberghieri
ent_with_no_1_2_star_hotel = set()
ent_with_3_4_star_hotel = set()
ent_with_5_star_hotel = set()
root = ET.parse("tndata/EserciziAlberghieri.xml").getroot()
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
df = pd.read_csv("tndata/bus_stops.csv", sep=",")
ent_with_bus_stop = set(map(sanitize,
                            [s.split("-")[0] for s in df.stop_name.unique()]))

# Read the agritur data provide by:
#   http://dati.trentino.it/dataset/agriturismi-del-trentino
df = pd.read_csv("tndata/elenco_agritur_attivi.csv", sep=";", engine="python")
ent_with_agritur = set(map(sanitize, df.sede_impresa_agricola.unique()))

# Read the museum data provided by:
#   https://it.wikipedia.org/wiki/Musei_del_Trentino-Alto_Adige
df = pd.read_csv("tndata/musei.csv", sep=",")
ent_with_museum = set(map(sanitize, df.ent.unique()))

# Read the osterie data provided by:
#   http://dati.trentino.it/dataset/osterie-tipiche-trentine
df = pd.read_csv("tndata/osterietipiche2013.csv", sep=";")
ent_with_osteria = set(map(sanitize,
                           [s.split("-")[0] for s in df.Comune.unique()]))

# Read the botteghe data provided by:
#   http://dati.trentino.it/dataset/botteghe-storiche-del-trentino
df = pd.read_csv("tndata/albobotteghestorichetrentinoluglio2016.csv", sep=",", engine="python")
ent_with_bottega = set(map(sanitize, df.Comune.unique()))

# Read the hospital data provided by:
#   http://dati.trentino.it/dataset/strutture-di-ricovero-e-cura
df = pd.read_csv("tndata/ospedali.csv", sep=",")
ent_with_ospedale = set(map(sanitize, df.COMUNE.unique()))

# Read the pharmacy data provided by:
#   http://dati.trentino.it/dataset/farmacie-pat
df = pd.read_csv("tndata/farmacie.csv", sep=",")
ent_with_farmacia = set(map(sanitize, df.COMUNE.unique()))

# Read the para-pharmacy data provided by:
#   http://dati.trentino.it/dataset/parafarmacie-pat
df = pd.read_csv("tndata/parafarmacie.csv", sep=",")
ent_with_parafarmacia = set(map(sanitize, df.COMUNE.unique()))

# Read the strutture sanitarie data provided by:
#   http://dati.trentino.it/dataset/strutture-sanitarie-dell-azienda-sanitaria-e-convenzionate
df = pd.read_csv("tndata/strutture_sanitarie.csv", sep=",")
ent_with_strutsan = set(map(sanitize, df.COMUNE.unique()))

# Read the strutture riabilitazione data provided by:
#   http://dati.trentino.it/dataset/strutture-di-riabilitazione
df = pd.read_csv("tndata/strutture_riabilitazione.csv", sep=",")
ent_with_strutriab = set(map(sanitize, df.COMUNE.unique()))

# Read the guardia medica data provided by:
#   http://dati.trentino.it/dataset/continuita-assitenziale-ex-gm
df = pd.read_csv("tndata/continuita_assistenziale.csv", sep=",")
ent_with_contassist = set(map(sanitize, df.COMUNE.unique()))

# Read the library data provided by:
#   http://dati.trentino.it/dataset/biblioteche-ai-censimenti
ent_with_library = set()
for line in open("biblioteche.xml", "rb"):
    line = line.decode().strip()
    if "<comune>" in line:
        ent = line.split(">")[1].split("<")[0]
        ent_with_library.add(sanitize(ent))


# Build the location->activity map
ent_with_activities = [
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
    ent_with_pprelievo,
]

num_locations = len(ents)
num_activities = len(ent_with_activities)

location_activities = np.zeros((num_locations + 1, num_activities), dtype=int)
for i, ent in enumerate(ents):
    for j, ent_with_activity in enumerate(ent_with_activities):
        if ent in ent_with_activity:
            location_activities[i, j] = 1

location_cost = np.hstack([
    np.random.randint(1, 10, size=num_locations),
    [0],
])

half = np.random.randint(1, MAX_TRAVEL_TIME + 1,
                         size=(num_locations, num_locations))
travel_time = np.rint((half + half.T) / 2).astype(int)
np.fill_diagonal(travel_time, 0)

num_regions = np.random.randint(2, num_locations)
regions = [np.random.randint(1, num_regions) for _ in range(num_locations)]

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
