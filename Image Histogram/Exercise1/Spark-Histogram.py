import cv2
import numpy as np
import pylab as plt
from pyspark.sql import SparkSession
from pyspark import SparkContext 

spark = SparkSession.builder.appName("Spark Histograms").getOrCreate()
sc = spark.sparkContext

Histogram_array = np.zeros(256,int)         # For Grayscale image

Histogram_array_R = np.zeros(256,int)       # Stores histogram values for Red scale in colored image 
Histogram_array_G = np.zeros(256,int)       # Stores histogram values for Green scale in colored image
Histogram_array_B = np.zeros(256,int)       # Stores histogram values for Blue scale in colored image

Histogram_list = [Histogram_array_B,Histogram_array_G,Histogram_array_R]

################# For Colored Image ##################

color_img = cv2.imread('/home/khalid/Documents/Homework08/2048.jpg')

single_array_B = np.ravel(color_img[:,:,0])     # Get the Blue pixels from the image and convert them into a 1d array using ravel
single_array_G = np.ravel(color_img[:,:,1])     # Get the Green pixels from the image and convert them into a 1d array using ravel
single_array_R = np.ravel(color_img[:,:,2])     # Get the Red pixels from the image and convert them into a 1d array using ravel

PixelIntensity_list = [single_array_B,single_array_G,single_array_R]        # Put them in a list for easier access

for items in range(3):

    lines = sc.parallelize(PixelIntensity_list[items])      # Parallelize the array

    final = lines.map(lambda x:(x,1))                       # Create a key-value pair for every value of the array

    seqOp = (lambda key,value:(key+value))                  # Add the key with the value

    combOp = (lambda key,value : (key+value))               # Again add and combine the values

    kk = final.aggregateByKey(0,seqOp,combOp)               # Perform aggregation 

    iterable_list = kk.collect()                            # Collect the results in a list

    for i in range(len(iterable_list)):
        Histogram_list[items][iterable_list[i][0]] += iterable_list[i][1]       # Substitute the value in the histogram array
    
plt.plot(np.arange(0,256),Histogram_list[2],color='R',label = "Red Pixel")
plt.plot(np.arange(0,256),Histogram_list[1],color='G', label = "Green Pixel")
plt.plot(np.arange(0,256),Histogram_list[0],color='B' , label = "Blue Pixel")
plt.ylabel('')
plt.xlabel('')
plt.legend()
plt.show()

################# For Gray Scale Image ####################

gray_img = cv2.imread('/home/khalid/Documents/Homework08/2048.jpg',cv2.IMREAD_GRAYSCALE)

single_array = np.ravel(gray_img)

lines = sc.parallelize(single_array)

final = lines.map(lambda x:(x,1))

seqOp = (lambda key,value:(key+value))

combOp = (lambda key,value : (key+value))

kk = final.aggregateByKey(0,seqOp,combOp)

iterable_list = kk.collect()

for i in range(len(iterable_list)):
    Histogram_array[iterable_list[i][0]] += iterable_list[i][1]   
    
plt.plot(np.arange(0,256),Histogram_array,color='Gray',label = "GrayScale")
        
plt.ylabel('')
plt.xlabel('')
plt.legend()
plt.show()

