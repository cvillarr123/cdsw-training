
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("deploy").getOrCreate()



rides = spark.read.parquet("/duocar/joined/")


# remover las carreras canceladas:
from pyspark.ml.feature import SQLTransformer
filterer = SQLTransformer(statement="select * from __THIS__ where cancelled = 0")

#  `star_rating` cambiar entrada de evaluacion de la carrera :
converter = SQLTransformer(statement="select *, cast(star_rating as double) as star_rating_double from __THIS__")

# binarizar  `star_rating_double`:
from pyspark.ml.feature import Binarizer
binarizer = Binarizer(inputCol="star_rating_double", outputCol="five_star_rating", threshold=4.5)

# extraer la variable  `reviewed` feature:
extractor = SQLTransformer(statement="select *, review is not null as reviewed from __THIS__")

# ensamblar las entradas:
from pyspark.ml.feature import VectorAssembler
selected = ["reviewed"]
assembler = VectorAssembler(inputCols=selected, outputCol="features")

# especificar el modelo:
from pyspark.ml.classification import DecisionTreeClassifier
classifier = DecisionTreeClassifier(featuresCol="features", labelCol="five_star_rating")

# especificar el pipeline:
from pyspark.ml import Pipeline
stages = [filterer, converter, binarizer, extractor, assembler, classifier]
pipeline = Pipeline(stages=stages)


# ## Save and load the machine learning pipeline

# guardar la instancia del  `Pipeline` HDFS:
pipeline.write().overwrite().save("models/pipeline")



# si no queremos sobreescribirlo:
#```python
#pipeline.save("models/pipeline")
#```

# leer el pipeline desde el hdfs  :
pipeline_loaded = Pipeline.read().load("models/pipeline")

# se puede usar esto otro método:
#```python
#pipeline_loaded = Pipeline.load("models/pipeline")
#```


# ## entrenar el modelo

pipeline_model = pipeline.fit(rides)


# ## guardar el modelo 

# guardar el pipeline model en  HDFS:
pipeline_model.write().overwrite().save("models/pipeline_model")



# leer el modelo desde el directorio en hdfs :
from pyspark.ml import PipelineModel
pipeline_model_loaded = PipelineModel.read().load("models/pipeline_model")


# ## Examinar y evaluar el algoritmo de clasificacion

# extraer el modelo desde el stage 5:
classifier_model = pipeline_model.stages[5]
type(classifier_model)

# usar el atributo `toDebugString` para imprimir el arbol de clasificacion :
print(classifier_model.toDebugString)

# usar el metodo  `transform` para aplicar el  pipeline model a un DataFrame para predecir  Dataframe --> DataFrame:
classified = pipeline_model.transform(rides)

# usar el metodo persist para dejar en memoria cache el Dataframe :
classified.persist()

# revisar el  DataFrame  classified:
classified.printSchema()

classified.select("review", "reviewed", "features").show(10)

classified.select("star_rating", "star_rating_double", "five_star_rating").show(10)

classified.select("probability", "prediction", "five_star_rating").show(10, truncate=False)

# calcular la matriz de confusion:
classified \
  .crosstab("prediction", "five_star_rating") \
  .orderBy("prediction_five_star_rating") \
  .show()

# extraer los aciertos la precision (sobre las 5 estrellas ):
from pyspark.sql.functions import col
classified.filter(1.0 == col("five_star_rating")).count() / \
float(classified.count())

# Calcular la precision del arbol de decision :
classified.filter(col("prediction") == col("five_star_rating")).count() / \
float(classified.count())

# Unpersist the classified DataFrame:
classified.unpersist()

# cerrar:
spark.stop()
