from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.ml.feature import IndexToString
from pyspark.sql import SparkSession
from pyspark import SparkContext 
from pyspark.sql import SQLContext

spark = SparkSession.builder.appName("Naive Bayes").getOrCreate()
sc = spark.sparkContext


sqlContext = SQLContext(sc)

def NaiveBayesPipeline(Data):

    training_data, testing_data = Data.randomSplit([0.8, 0.2])              # Split the data into training and testing data
    
    categoryIndexer = StringIndexer(inputCol="Decision", outputCol="label") # Perfrom indexing on the Decision field

    tokenizer = Tokenizer(inputCol="Message", outputCol="words")            # Tokenize the values

    hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=10000)    # Perform TF based on Hashing TF

    naiveBayes = NaiveBayes(smoothing=1.0, modelType="multinomial")         # Create a Naive Bayes model

    pipeline = Pipeline(stages=[categoryIndexer, tokenizer, hashingTF, naiveBayes])     # Create a pipeline for the data

    model = pipeline.fit(training_data)         # Fit the training data in the pipeline
    result = model.transform(testing_data)      # Check the performance of the model on the testing data

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")        # Evaluate the model
    metric = evaluator.evaluate(result)     

    print "\n\n\n\n\nF1 metric = %g\n\n" % metric           # Print the F1 score of the model

if __name__=="__main__":
    
    # Load the data then split it by the first comma occurence
    data = sc.textFile("/home/khalid/Documents/Homework08/exercise_8_first_term_students/Spam.csv").map(lambda line: line.split(",",1))
        
    # Create a data frame from the above data
    newDF = sqlContext.createDataFrame(data, ["Decision", "Message"])

    NaiveBayesPipeline(newDF)    

