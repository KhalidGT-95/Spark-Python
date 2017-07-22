from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import sys
from random import random
from operator import add
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression

# Get the spark object
spark = SparkSession.builder.appName("DataFrame").getOrCreate()

# Get the spark context from the spark object
sc = spark.sparkContext
     
sc.setLogLevel("ERROR")

# Read the json file for input    
df = spark.read.json('/home/khalid/exercise_7_for_first_term_students/tweets.json')

# Declare a Logistic Regression Model
lr = LogisticRegression(maxIter=10, regParam=0.001)

# Tokenize the data
tokenizer = Tokenizer(inputCol="content", outputCol="words")
wordsData = tokenizer.transform(df)

# Use HashingTF function to calculate TF function
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# Use idf to get IDF function
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
idfModel = idf.fit(featurizedData)

# Transform the model
rescaledData = idfModel.transform(featurizedData)

# The pipeline pipes the result of one function to the input of another
# In the end we give the output TFIDF to a Logistic Regression model for prediction
# In this way we can achieve Machine Learning using Spark Pipelining
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])

rescaledData.select("content","features").show()
