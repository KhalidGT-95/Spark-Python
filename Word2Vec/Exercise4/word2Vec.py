from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import sys
from random import random
from operator import add
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Word2Vec


spark = SparkSession.builder.appName("DataFrame").getOrCreate()
         
sc = spark.sparkContext
     
sc.setLogLevel("ERROR")
    
df = spark.read.json('/home/khalid/exercise_7_for_first_term_students/tweets.json')

tokenizer = Tokenizer(inputCol="content", outputCol="words")
wordsData = tokenizer.transform(df)

new_words = wordsData.select('words')			# Taken from previous exercise	

word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="words", outputCol="result")
model = word2Vec.fit(new_words)

result = model.transform(new_words)

result.select('result').show()		# Show top 1 rows

