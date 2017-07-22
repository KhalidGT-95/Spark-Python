from pyspark.sql.types import DoubleType
from pyspark.sql.functions import UserDefinedFunction
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark import SparkContext 
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName("Decision Tree Classifier").getOrCreate()
sc = spark.sparkContext


sqlContext = SQLContext(sc)

def LabellingOfData(data):
    return data.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))

# Perform prediction on test data
def getPredictionsFromLabels(model, test_data):
    predictions = model.predict(test_data.map(lambda r: r.features))
    return predictions.zip(test_data.map(lambda r: r.label))

## This function prints the final results based on the training of the model
def printFinalResultMetrics(predictions_and_labels):
    metrics = MulticlassMetrics(predictions_and_labels)
    print '\n'
    print 'Precision of Setosa ', metrics.precision(1)
    print 'Precision of Versicolor', metrics.precision(2)
    print 'Precision of Virginica', metrics.precision(3)
    print '\n'
    print 'Recall of Setosa    ', metrics.recall(1)
    print 'Recall of Versicolor   ', metrics.recall(2)
    print 'Recall of Virginica   ', metrics.recall(3)
    
    print '\n'
    print 'F-1 Score         ', metrics.fMeasure()
    print '\n\n'
    print 'Confusion Matrix\n', metrics.confusionMatrix().toArray()
    
    print '\n\n' 
    return
    

if __name__=="__main__":    
    csv_data = sqlContext.read.load('/home/khalid/Documents/Homework08/exercise_8_first_term_students/iris.csv',        # Read the csv file
                              format='com.databricks.spark.csv', 
                              header='true', 
                              inferSchema='true')
                              
                              
    ternary_map = {'setosa': 1, 'versicolor':2 , 'virginica':3}     # Create a dictionary for a key-value map which will be used later to convert string value to number

    ConvertColumnValuesToNumbers = UserDefinedFunction(lambda k: ternary_map[k], IntegerType())     # Create a function which will convert string to numbers using the above lookup dictionary

    csv_data = csv_data.withColumn('Species',ConvertColumnValuesToNumbers(csv_data['Species']))     # Convert every string to an Integer value based on the above dictionary


    training_data, testing_data = LabellingOfData(csv_data).randomSplit([0.8, 0.2])                 # Split the data into training and tesing data

    DecisionTreeModel = DecisionTree.trainClassifier(training_data, numClasses=4, maxDepth=3,       # Train the Decision Tree model
                                         categoricalFeaturesInfo={},
                                         impurity='gini', maxBins=16)

    ## Increasing the maxDepth increases the f1 score

    print DecisionTreeModel.toDebugString()     # Print the Decision tree


    predictionsAndLabels = getPredictionsFromLabels(DecisionTreeModel, testing_data)

    printFinalResultMetrics(predictionsAndLabels)

