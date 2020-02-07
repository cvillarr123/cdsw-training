

# ## Introducción

# * Un algoritmo de clasificación es un algoritmo de aprendizaje supervisado
# * Las entradas se denominan * características *
# * La salida se llama * etiqueta *

# * Un modelo de clasificación proporciona una predicción de una etiqueta categórica
# * Clasificación binaria: dos categorías
# * Clasificación multiclase: tres o más categorías

# * Spark MLlib proporciona varios algoritmos de clasificación:
#   * Logistic Regression (with Elastic Net, Lasso, and Ridge Regression)
#   * Decision Tree
#   * Random Forest
#   * Gradient-Boosted Trees
#   * Multilayer Perceptron (Neural Network)
#   * Linear Support Vector Machine (SVM)
#   * Naive Bayes

# * Spark MLlib también proporciona un meta-algoritmo para construir multiclase
# modelos de clasificación de modelos de clasificación binarios:
# * Uno contra el descanso

# * Spark MLlib requiere que las características se ensamblen en una columna de vector de dobles

# * Spark MLlib requiere que la etiqueta sea un índice basado en cero


# ## Scenario

# En este módulo modelaremos la calificación de estrellas de un viaje en función de
# varios atributos del viaje. En lugar de tratar la calificación de estrellas en su
# forma original, crearemos una etiqueta binaria que sea verdadera si la calificación es
# cinco estrellas y falso de lo contrario.  Se va a usar [logistic
# regression](https://en.wikipedia.org/wiki/Logistic_regression) para construir un
# modelo de clasificación binaria. El flujo de trabajo general será similar para otros
# algoritmos de clasificación, aunque los detalles particulares variarán.


# ## Setup

from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql.functions import col


# ## Start a SparkSession

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("classify").getOrCreate()


# ## Leer los datos

# Leer los datos unificados en un solo dataset desde el HDFS:
rides = spark.read.parquet("/duocar/joined/")


# ## Preprocesando los datos

# Una carrera cancelada no tiene registros de puntuacion Usar
# [SQLTransformer](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.SQLTransformer)
# para filtrar las carreras canceladas:
from pyspark.ml.feature import SQLTransformer
filterer = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE cancelled == 0")
filtered = filterer.transform(rides)

# **Note:** `__THIS__` es un is a marcador de posicion para pasar el nombre del dataframe en el 
# metodo de `transform` .


# ## Generar las etiquetas

# Se puede tratar al campo  `star_rating` como una etiqueta numerica o una 
# etiqueta de categoria ordenada:
filtered.groupBy("star_rating").count().orderBy("star_rating").show()

# Para predecir si es buena o mala la carrera se procedera a establecer un umbral 
# que distinguira entre una carrera mala de buena esto es las menores a 4.5 son malas
# usando el metodo
# [Binarizer](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Binarizer)
# para crear la categoria de etiqueta binaria:
from pyspark.ml.feature import Binarizer
converted = filtered.withColumn("star_rating", col("star_rating").cast("double"))
binarizer = Binarizer(inputCol="star_rating", outputCol="high_rating", threshold = 4.5)
labeled = binarizer.transform(converted)
labeled.crosstab("star_rating", "high_rating").show()

# **Note:** `Binarizer` este campo no es un valor entero, por lo que se tuvo que convertir a doble.


# ## Extraer, transformar y seleccionar las caracteristicas  Extract, transform, and select features

# Crear una funcion para explorar los datos:
def explore(df, feature, label, plot=True):
  from pyspark.sql.functions import count, mean
  aggregated = df.groupby(feature).agg(count(label), mean(label)).orderBy(feature)
  aggregated.show()
  if plot == True:
    pdf = aggregated.toPandas()
    pdf.plot.bar(x=pdf.columns[0], y=pdf.columns[2], capsize=5)

# **Feature 1:** El pasajero califico la carrera? Did the rider review the ride?
engineered1 = labeled.withColumn("reviewed", col("review").isNotNull().cast("int"))
explore(engineered1, "reviewed", "high_rating")

# **Note:** EL  `avg(high_rating)` da la fracción observada de una alta calificación.

# **Feature 2:** ¿Importa el año del vehículo?
explore(labeled, "vehicle_year", "high_rating")

# **Note:** Es más probable que el pasajero otorgue una calificación alta cuando el automóvil está
# nuevo.  Trataremos esta variable como una característica continua.

# **Feature 3:** ¿Qué pasa con el color del vehículo?
explore(labeled, "vehicle_color", "high_rating")

# **Note:** Es más probable que el conductor otorgue una calificación alta si el automóvil está
# negro y es menos probable que otorgue una calificación alta si el automóvil es amarillo.



# Los algoritmos de clasificación en Spark MLlib no aceptan categóricos
# características en este formulario, así que vamos a convertir `vehicle_color` a un conjunto 
# de variables dummy. Primero se usa 
# [StringIndexer](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.StringIndexer)
# para convertir los códigos de cadena a codigos numéricos:
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="vehicle_color", outputCol="vehicle_color_ix")
indexer_model = indexer.fit(engineered1)
list(enumerate(indexer_model.labels))
indexed = indexer_model.transform(engineered1)
indexed.select("vehicle_color", "vehicle_color_ix").show(5)

# Luego usamos la libreria
# [OneHotEncoder](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.OneHotEncoder)
# para generar variables dummy :
from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(inputCol="vehicle_color_ix", outputCol="vehicle_color_cd")
encoded = encoder.transform(indexed)
encoded.select("vehicle_color", "vehicle_color_ix", "vehicle_color_cd").show(5)

# **Note:** `vehicle_color_cd` es guardado como un  `SparseVector`.

# Se puede  (manualmente) seleccionar las caracteristicas y etiquetas :
selected = encoded.select("reviewed", "vehicle_year", "vehicle_color_cd", "star_rating", "high_rating")
features = ["reviewed", "vehicle_year", "vehicle_color_cd"]

# MLIB espera que las caracteristicas sean guardadas en una sola columna
# asi que se usa la clase 
# [VectorAssembler](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler)
# para guardarlo en una columna de vector:
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=features, outputCol="features")
assembled = assembler.transform(selected)
assembled.head(5)

# **Note:** `features` es guardado como  `SparseVector`.

# guardar los datos en un archivo :
assembled.write.parquet("data/modeling_data", mode="overwrite")

# **Note:** se guardara en nuestro directorio HDFS.


# ## Crear el entrenamiento y el test


(train, test) = assembled.randomSplit([0.7, 0.3], 12345)

# **Important:**  Los pesos son doubles.


# ## Especificar un modelo logistico

#
# [LogisticRegression](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression)
# es la clase para especificar un modelo logistico :
from pyspark.ml.classification import LogisticRegression
log_reg = LogisticRegression(featuresCol="features", labelCol="high_rating")

# Usar el metodo  `explainParams` para ver la lista de los hyperparametros:
print(log_reg.explainParams())


# ## Entrenar el modelo

# Metodo fit
log_reg_model = log_reg.fit(train)

# El resultado es una instancia de la clase
# [LogisticRegressionModel](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegressionModel)
# class:
type(log_reg_model)


# ## Examinar los parametros del modelo

# como atributos del modelo `intercept` and `coefficients` :
log_reg_model.intercept
log_reg_model.coefficients

# El atributo  `summary` 
# [BinaryLogisticRegressionTrainingSummary](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.BinaryLogisticRegressionTrainingSummary)
# :
type(log_reg_model.summary)

# Se puede consultar las iteraciones y el historial:
log_reg_model.summary.totalIterations
log_reg_model.summary.objectiveHistory

# and plot it too:
#def plot_iterations(summary):
#  plt.plot(summary.objectiveHistory)
#  plt.title("Training Summary")
#  plt.xlabel("Iteration")
#  plt.ylabel("Objective Function")
#  plt.show()

# plot_iterations(log_reg_model.summary)

# Se puede ver el perfomance del modelo en este caso el area bajo la curva :
log_reg_model.summary.areaUnderROC

# y dibujar dicha area:
log_reg_model.summary.roc.show(5)

# def plot_roc_curve(summary):
  # roc_curve = summary.roc.toPandas()
  # plt.plot(roc_curve["FPR"], roc_curve["FPR"], "k")
  # plt.plot(roc_curve["FPR"], roc_curve["TPR"])
  # plt.title("ROC Area: %s" % summary.areaUnderROC)
  # plt.xlabel("False Positive Rate")
  # plt.ylabel("True Positive Rate")
  # plt.show()

# plot_roc_curve(log_reg_model.summary)


# ## Evaluar el modelo con el dataset de test


# Hemos estado evaluando el rendimiento del modelo en el DataFrame del tren. Nosotros
# realmente quiero evaluarlo en la prueba DataFrame.

# **Metodo 1:** usando el metodo  `evaluate` de la claase `LogisticRegressionModel` 

test_summary = log_reg_model.evaluate(test)

# el resultado es una instancia de la clase
# [BinaryLogisticRegressionSummary](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.BinaryLogisticRegressionSummary)
# class:
type(test_summary)

# Tiene atributos similares a los de la clase
# `BinaryLogisticRegressionTrainingSummary` :
test_summary.areaUnderROC
plot_roc_curve(test_summary)

# ****Metodo 2:** usar el metodo `evaluate` de la clase
# [BinaryClassificationEvaluator](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.BinaryClassificationEvaluator)
# 


# Generar predicciones en el DataFrame de prueba:
test_with_prediction = log_reg_model.transform(test)
test_with_prediction.show(5)

test_summary_pred = log_reg_model.evaluate(test_with_prediction)
plot_roc_curve(test_summary)


# ** Nota: ** El DataFrame resultante incluye tres tipos de predicciones. los
# `rawPrediction` es un vector de log-odds,` prediction` es un vector o
# probabilidades `prediction` es la clase predicha basada en la probabilidad
# vector.

# Crear una instancia de la clase `BinaryClassificationEvaluator` :
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="high_rating", metricName="areaUnderROC")
print(evaluator.explainParams())
evaluator.evaluate(test_with_prediction)



# Evaluar usando la metrica :
evaluator.setMetricName("areaUnderPR").evaluate(test_with_prediction)




# ## References

# [Spark Documentation - Classification and regression](https://spark.apache.org/docs/latest/ml-classification-regression.html)

# [Spark Python API - pyspark.ml.feature module](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.feature)

# [Spark Python API - pyspark.ml.classification module](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.classification)

# [Spark Python API - pyspark.ml.evaluation module](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.evaluation)


# ## parar la sesion

spark.stop()
