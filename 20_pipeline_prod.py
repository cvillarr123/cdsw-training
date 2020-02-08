


# ## Ambiente


from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("pipeline").getOrCreate()




# Read the enhanced ride data from HDFS:
rides = spark.read.parquet("/duocar/joined-all/")




# Create the train and test DataFrames *before* specifying the pipeline:
(train, test) = rides.randomSplit([0.7, 0.3], 12345)


# ## Specify the pipeline stages

# A *Pipeline* es una secuencia de estados que se implementan para ingenieria de datos
# o para un flujo de trabajo para machine learning



# Filtrado:
from pyspark.ml.feature import SQLTransformer
filterer = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE cancelled == 0")

# Generar los inputs:
extractor = SQLTransformer(statement="SELECT *, review IS NOT NULL AS reviewed FROM __THIS__")

# Indexar el campo  `vehicle_color`:
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="vehicle_color", outputCol="vehicle_color_indexed")

# crear un dummy para la categorica de  `vehicle_color_indexed`:
from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(inputCol="vehicle_color_indexed", outputCol="vehicle_color_encoded")

# seleccionar los features 
from pyspark.ml.feature import VectorAssembler
features = ["reviewed", "vehicle_year", "vehicle_color_encoded", "CloudCover"]
assembler = VectorAssembler(inputCols=features, outputCol="features")

# especificar el estimador (i.e., classification algorithm):
from pyspark.ml.classification import RandomForestClassifier
classifier = RandomForestClassifier(featuresCol="features", labelCol="star_rating")
print(classifier.explainParams())

# espeficar los valores en el grid de hiperparametros:
from pyspark.ml.tuning import ParamGridBuilder
maxDepthList = [5, 10, 20]
numTreesList = [20, 50, 100]
subsamplingRateList = [0.5, 1.0]
paramGrid = ParamGridBuilder() \
  .addGrid(classifier.maxDepth, maxDepthList) \
  .addGrid(classifier.numTrees, numTreesList) \
  .addGrid(classifier.subsamplingRate, subsamplingRateList) \
  .build()

# especificar el evaluador:
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="star_rating", metricName="accuracy")

# **Note:** We are treating `star_rating` as a multiclass label.

# especificar el validador:
from pyspark.ml.tuning import TrainValidationSplit
validator = TrainValidationSplit(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator)


# ## especificar y armar el pipeline

# A
# [Pipeline](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline)
# este en si mismo es un  `Estimator`:
from pyspark.ml import Pipeline
stages = [filterer, extractor, indexer, encoder, assembler, validator]
pipeline = Pipeline(stages=stages)


# ## entrenar el modelo

# el metodo fit crea un objeto 
# [PipelineModel](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.PipelineModel),
# que es por debajo un `Transformer`:
#%time 

pipeline_model = pipeline.fit(train)


# ## Query the PipelineModel

# acceder a las instancias de cada `PipelineModel` usando el  atributo `stages` :
pipeline_model.stages

# se puede acceder de forma individual:
indexer_model = pipeline_model.stages[2]
indexer_model.labels

# el mejor modelo en este caso fue un RandomForest del tipo
# [RandomForestClassificationModel](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassificationModel)
# class:
validator_model = pipeline_model.stages[5]
type(validator_model.bestModel)

# obtener los mejores hiperparametros:
validator_model.bestModel._java_obj.getMaxDepth()
validator_model.bestModel.getNumTrees
validator_model.bestModel._java_obj.getSubsamplingRate()

# acceder a los valores de  `maxDepth` and `subsamplingRate`
# desde un objeto Java object.

# Plot feature importances:
def plot_feature_importances(fi):
  fi_array = fi.toArray()
  plt.figure()
  sns.barplot(range(len(fi_array)), fi_array)
  plt.title("Feature Importances")
  plt.xlabel("Feature")
  plt.ylabel("Importance")
plot_feature_importances(validator_model.bestModel.featureImportances)


# ## predecir usando el pipeline

# usando el metodo transform:
classified = pipeline_model.transform(test)
classified.printSchema()

# obtener la matriz de confusion:
classified \
  .crosstab("prediction", "star_rating") \
  .orderBy("prediction_star_rating") \
  .show()

  
  
pipeline_model.stages[-1].bestModel.featureImportances  

classified.schema["features"].metadata

classified.schema["features"].metadata["ml_attr"]["attrs"]


  
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        print("tipo de i")
        print (type(i))
        print("contenido de i")
        print (i)
        print("contenido de una iteracion")
        print(dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i])
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    print("tipo objeto list extract")
    print(type(list_extract))
    print("contenido list extract")
    print(list_extract)
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))


def plot_feature_importances_v2(fi):
  plt.figure()
  grafico=sns.barplot(x="name",y="score",data=fi)
  grafico.set_xticklabels(grafico.get_xticklabels(),rotation=45,horizontalalignment='right')
  plt.title("Feature Importances")
  plt.xlabel("Feature")
  plt.ylabel("Importance")
  

  
  
features_imporantes=ExtractFeatureImp(pipeline_model.stages[-1].bestModel.featureImportances, classified, "features").head(10)  


plot_feature_importances_v2(features_imporantes)
  
# poner como base de la prediccion al valor 5 estrellas  :
from pyspark.sql.functions import lit
classified_with_baseline = classified.withColumn("prediction_baseline", lit(5.0))

# evaluar la linea base del modelo y el modelo en si :
evaluator = MulticlassClassificationEvaluator(labelCol="star_rating", metricName="accuracy")
evaluator.setPredictionCol("prediction_baseline").evaluate(classified_with_baseline)
evaluator.setPredictionCol("prediction").evaluate(classified_with_baseline)




# ## Stop the SparkSession

spark.stop()
