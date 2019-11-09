# OS
import os
import sys

# Base
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Viz
from matplotlib.cm import viridis
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_hex
from matplotlib.collections import LineCollection


import networkx as nx
import json
# import random as random


# Geo
import geojson
# import pysal as ps
import geopandas as gpd
# from pysal.contrib.viz import mapping as maps
import esda
import libpysal as lps
import mapclassify as mc

# raster images
# import rasterio
# from scipy import ndimage

# database connections
import psycopg2 as pg
import sqlalchemy
from sqlalchemy import create_engine
from geoalchemy2 import Geometry, WKTElement
# from sqlalchemy import *

# Display
from IPython.display import GeoJSON
from IPython.display import JSON
# from IPython.display import HTML
# from IPython.display import Latex
# from IPython.display import Math

import folium
import googlemaps
import gmaps