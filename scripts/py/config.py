'''
Imports
-------------------------------------------------------------------------------------------
'''
# Python
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 3rd Party


# Local 


'''
Set program parameters
-------------------------------------------------------------------------------------------
'''

# Set tweet storage folder
def GetAQFolder():
    return '../data/air_quality'

def GetCamFolder():
    return '../data/cameras'

def GetNetFolder():
    return '../data/network'

def GetModelsFolder():
    return '../data/models'

def GetCensusFolder():
    return '../data/census'

def GetGeoFolder():
    return '../data/geography'

def GetDetectionFolder():
    return '../data/detection'

# Set model storage folder
def GetOutputFolder():
    return '../data/output_data'

'''
Helper functions
-------------------------------------------------------------------------------------------
'''

def GetFileString(month,day,hour):
    return '/M'+str(month)+"/D"+str(day)+"/H"+str(hour)

def GetImageRoot(month,day,hour):
    return GetCamFolder()+GetFileString(month,day,hour)

# Get filename for a set of tweets.Removed default to 0
def GetImageFolderName(month,day,hour,camID):       
    return GetCamFolder()+GetFileString(month,day,hour)+'/C'+str(camID)



# Get a filename for given 3 digit combo. Uses stub for all. Add as default stubto allow arg.



# Write json to file
def WriteJSON(obj,filename):
    try:
        with open(filename, 'w') as outfile:
            obj_json = json.dumps(obj, sort_keys=True, indent=4,default=str)
            outfile.write(obj_json)
    except Exception as e:
        print(e, file=sys.stderr)
        print('File not written.')

# Read and return json object from file. If none, return empty object.
def ReadJSON(filename):
    try: 
        with open(filename, 'r') as infile:
            obj = json.load(infile)
    except Exception as e: 
        obj = [] 
    return obj

# Write df to csv
def WriteCSV(data,filename):
    stub = GetOutputFolder()
    filestring = stub+'/'+filename
    with open(filestring,'w') as outfile:
        data.to_csv(outfile)

# Read csv to df
def ReadCSV(filename):
    stub = GetAQFolder()
    filestring = stub+'/'+filename
    data = pd.read_csv(filestring,header=0,encoding = "ISO-8859-1")
    # ,index_col='Unnamed: 0'
    return data

