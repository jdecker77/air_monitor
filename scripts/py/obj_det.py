'''
Import py libraries
'''

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image,ImageDraw

# Object detection imports
# from utils import label_map_util
# from utils import visualization_utils as vis_util

from IPython.display import Image as IMG

import config


'''
Load image paths
'''

def GetImagePaths(month,day,hour,calls,cams):

    # Loop through calls per hour
    image_paths = []
    
    for call in calls[:]:
        for camID in cams:
            img_folder = config.GetImageFolderName(month,day,hour,camID)
            img_path = img_folder+'/'+str(camID)+'_'+str(call)+'.jpg'
            image_paths.append(img_path)
    
    return image_paths


# In[4]:


'''
Get paths to storage - inside each hour 
'''

def GetStorage(month,day,hour):
    
    # Set storage based on month/day/hour
    storage = config.GetImageRoot(month,day,hour)

    # Set filenames for storage
    img_results_file = storage+'/image_results.csv'
    summary_results_file = storage+'/traffic_measurement.csv'
    
    return img_results_file,summary_results_file


# In[5]:


'''
Initialize summmary csv with header
'''

def InitSummaryCSV(filename,vehicles):
    
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        csv_line = 'ImagePath,NumDetections,'
        for v in vehicles:
            csv_line += str(v)+','
        writer.writerows([csv_line.split(',')])


# In[6]:


'''
Initialize image csv with header
'''

def InitImageCS(filename):

    # initialize .csv
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        csv_line = 'ImagePath,class,score,left, right, top, bottom'
        writer.writerows([csv_line.split(',')])


# In[7]:


'''
Load images as array
'''

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,3)).astype(np.uint8)


# In[8]:


'''
Initialize tensors
'''

def InitTensors(detection_graph):
    # Define input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent the level of confidence for each of the objects.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

    # Each class represents classification          
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of detections in image - more than just vehicles             
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    return image_tensor,detection_boxes,detection_scores,detection_classes,num_detections


# In[9]:


'''
Process results of obj det

1. Limit results by class & score
2. Save image level results
3. Collect and return summary level data
'''               

def ProcessResults(img_path,filename,targets,threshold,boxes,scores,classes,num):
    ubox = []
    uscore = []
    uclasses = []
    for i in range(0,len(classes)):
        for j in range(0,len(classes[i])):
            if classes[i][j] in targets and scores[i][j] >= threshold:
                # print(classes[i][j])

                # Process Image level results                             
                csv2_line = img_path+','
                ymin, xmin, ymax, xmax = boxes[i][j]

                img = Image.open(img_path)
                im_width, im_height = img.size
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,ymin * im_height, ymax * im_height)
                # (left, right, top, bottom) = (xmin, xmax,ymin, ymax)
                csv2_line += str(category_index[classes[i][j]]['name'])+','+str(scores[i][j])+','+str(int(left))+','+str(int(right))+','+str(int(top))+','+str(int(bottom))

                # Save image level results
                with open(filename, 'a') as f:
                    writer = csv.writer(f)
                    (name,classi,score,left, right, top, bottom) = csv2_line.split(',')
                    writer.writerows([csv2_line.split(',')])

                # Collect summary items limited by class & accuracy score
                ubox.append(boxes[i][j])
                uscore.append(scores[i][j])
                uclasses.append(classes[i][j])

    # Count of total cars     
    unum = len(uclasses)
    
    return ubox,uscore,uclasses,unum


# In[10]:


'''
Process and save summary level results
'''

def ProcessSummary(img_path,summary_results_file,targets,uclasses,unum):
    csv_line = img_path+','+str(unum)+','
    counts = np.unique(uclasses,return_counts=True)
    # print(counts)
    for i in targets:
        # print(list(counts[1]))
        if i in list(counts[0]):
            l = list(counts[0]).index(i)
            csv_line += str(counts[1][l])
            if i != 9:
                csv_line += ','
        else:
            csv_line += '0'
            if i != 9:
                csv_line += ','
#                 print(csv_line)
    
    # Save summary results
    with open(summary_results_file, 'a') as f:
        writer = csv.writer(f)
        (name,num,a,b,c,d,e,f,g,h) = csv_line.split(',')
        writer.writerows([csv_line.split(',')])


'''
Initialize Graph with frozen model or other
'''
def InitGraph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


'''
Object Detection
'''

def object_detection_function(image_paths,targets,category_index,threshold,img_results_file,summary_results_file):

    detection_graph = InitGraph()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            print('Starting session')
            
            image_tensor,detection_boxes,detection_scores,detection_classes,num_detections = InitTensors(detection_graph)

            # Loop through images
            for img_path in image_paths:
#                 print('Image:',img_path)

                # Load image 
                try:               
                    img = image = Image.open(img_path)
                    input_frame = load_image_into_numpy_array(img)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(input_frame, axis=0)

                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores,
                                                              detection_classes, num_detections],
                                                             feed_dict={image_tensor: image_np_expanded})         
                    
                    # Process results of obj det                 
                    ubox,uscore,uclasses,unum = ProcessResults(img_path,img_results_file,targets,threshold,boxes,scores,classes,num)    
    #                 print("Number detections",unum)
                
                    # Process summary results 
                    ProcessSummary(img_path,summary_results_file,targets,uclasses,unum)
                except Exception as e:
                    print('Error',e)
                    print('Error in OBJ Function. Skipping image',img_path)


# In[17]:


'''
Call OBJ DET here 
'''

def StartDetection(month,day,hour,calls,cams,targets,threshold):
    # Load image paths
    img_paths = GetImagePaths(month,day,hour,calls,cams)

    # Get output paths
    img_results_file,summary_results_file = GetStorage(month,day,hour)
    
    # Initialize output CSVs
    InitImageCS(img_results_file)
    # Below adds nums to csv - convert to classes
    InitSummaryCSV(summary_results_file,[category_index[i]['name'] for i in targets])

    object_detection_function(img_paths,targets,category_index,threshold,img_results_file,summary_results_file)


