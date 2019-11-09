import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString, shape, MultiPoint, box, Polygon, MultiLineString, mapping
from shapely.ops import linemerge


import config

'''
Get new graph from point

Use network_type or custom filter, custom_filter='["highway"~"cycleway"]'
'''

def GetNewGraph(net_type):
    
    point = (float(station.iloc[0]['LATITUDE']),float(station.iloc[0]['LONGITUDE']))

    gph = ox.graph_from_point(point,5000,distance_type='network',network_type=net_type,simplify=True,infrastructure='way["highway"]')
    gph = ox.remove_isolated_nodes(gph)
    
    return gph

'''
Save graph as both shapefile and graph
'''

def SaveGraph(H,filename):
    ox.save_graph_shapefile(H,filename=filename,folder=GetNetFolder())
    ox.save_graphml(H,filename=filename+'.graphml',folder=GetNetFolder())


# In[29]:




# In[9]:


'''
Load graph from file
'''
def LoadSavedGraph(filename):
    return ox.load_graphml(filename=filename,folder=GetNetFolder())


# In[20]:


'''
Convert graph to edge/nodes dataframes
'''

def GraphToFrames(H):
    return ox.graph_to_gdfs(H)


# In[23]:


'''
Add load to nodes
'''

def LoadNodeCounts(osmid):
#     print(osmid,type(osmid))
    ids = counts.osmid.tolist()
    if osmid in ids:
#         print(osmid,type(osmid))
        ix = ids.index(osmid)
        for col in cols:
#             print(nodes.loc[ix][col])
            nodes.at[ix,col] = int(counts.at[ix,col])


# In[25]:


'''
Convert edge/nodes dataframes to graph
'''

def FramesToGraph(edges,nodes):
    return ox.gdfs_to_graph(nodes, edges)


