


# ## Introduction

# Most machine learning algorithms have a set of user-specified parameters that
# govern the behavior of the algorithm.  These parameters are called
# *hyperparameters* to distinguish them from the model parameters such as the
# intercept and coefficients in linear and logistic regression.  In this module
# we show how to use grid search and cross validation in Spark MLlib to
# determine a reasonable regularization parameter for

# ## Introducción

# La mayoría de los algoritmos de aprendizaje automático tienen un conjunto de parámetros especificados por el usuario que
# gobierna el comportamiento del algoritmo. Estos parámetros se llaman
# * hiperparámetros * para distinguirlos de los parámetros del modelo, como el
# intercepción y coeficientes en regresión lineal y logística. En este modulo
# mostramos cómo usar la búsqueda de cuadrícula y la validación cruzada en Spark MLlib para
# determinar un parámetro de regularización razonable para [$l1$ lasso linear
# regression](https://en.wikipedia.org/wiki/Lasso_%28statistics%29).


# ## Setup

# 
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Crear una sesion de spark

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("hypertune").getOrCreate()


# ## Leer los datos

# **Important:**  Debio correrse el script `15_classify.py` antes de leer estos datos.

# Read the modeling data from HDFS:
rides = spark.read.parquet("data/modeling_data")
rides.show(5)


# ## Crear los datos de entrenamiento y test

(train, test) = rides.randomSplit([0.7, 0.3], 12345)


# ## Requerimeintos de afinamiento de hyper parametros 


# Necesitamos especificar cuatro componentes para realizar el ajuste de hiperparámetros usando
# busqueda de la malla:
# * Estimator (i.e. machine learning algorithm)
# * Hyperparameter grid
# * Evaluator
# * Validation method


# ## Specify the estimator

# En este ejemplo usaremos la regresion lineal lasso para nuestro estimador :
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol="features", labelCol="star_rating", elasticNetParam=1.0)

# Usar el metodo  `explainParams`  para ver la lista de los hiperparametros:
print(lr.explainParams())

# Configurar  `elasticNetParam=1.0` corresponde al modelo Lasso  $l1$ (lasso) de regresion linear 
# Queremos encontrar un valor razonable de ese parametro que esta en el objeto `regParam`.
# [Elastic_net](https://en.wikipedia.org/wiki/Elastic_net_regularization)

# ## Especificar un grid de parametros 
# 

# usar la clase para especificar el grid (malla) de hiperparametros
# [ParamGridBuilder](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder)
from pyspark.ml.tuning import ParamGridBuilder
regParamList = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
grid = ParamGridBuilder().addGrid(lr.regParam, regParamList).build()

# El objeto resultante es simplemente una lista de mapas de parámetros:
grid

# En lugar de especificar `elasticNetParam` en el constructor de la clase  `LinearRegression` , podemos especificarlo en nuestra cuadrícula:
grid = ParamGridBuilder().baseOn({lr.elasticNetParam: 1.0}).addGrid(lr.regParam, regParamList).build()
grid


# ## Especificar el evaluador

# In this case we will use
# [RegressionEvaluator](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator)
# como nuestro evaluador y especifique el error cuadrático medio como la métrica:
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="star_rating", metricName="rmse")


# ## Afinar los parametros usando  holdout cross-validation

# En muchos casos, holdout cross-validation will be sufficient. Usar la clase
# [TrainValidationSplit](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.TrainValidationSplit)
# para especificar  holdout cross-validation:
from pyspark.ml.tuning import TrainValidationSplit
tvs = TrainValidationSplit(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, trainRatio=0.75, seed=54321)

# material teorico:
# [TrainValidationSplit](https://es.wikipedia.org/wiki/Validacion_cruzada)

# Para cada combinacion de hiperparametros la regresion linear sera, 
# entrenada con un set aleatorio de 75% de registros para entrenamiento llenando el DataFrame `train` 
# y luego evaluado sobre el 25%. 

# usar el metodo  `fit` para encontrar el mejor conjunto de parametros:
%time tvs_model = tvs.fit(train)


# El resultado es una instancia de la clase 
# [TrainValidationSplitModel](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.TrainValidationSplitModel)
#:
type(tvs_model)

# Los resultados de validación cruzada se almacenan en el atributo `validationMetrics`:

tvs_model.validationMetrics

# Estos son los RMSE para cada conjunto de hiperparámetros. Más pequeño es mejor.

def plot_holdout_results(model):
  plt.plot(regParamList, model.validationMetrics)
  plt.title("Hyperparameter Tuning Results")
  plt.xlabel("Regularization Parameter")
  plt.ylabel("Validation Metric")
  plt.show()
plot_holdout_results(tvs_model)

# En este caso el atributo `bestModel` es una instancia de la clase de regresion lineal
# [LinearRegressionModel](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegressionModel)
# class:
type(tvs_model.bestModel)

# ** Nota: ** El modelo se vuelve a ejecutar en todo el conjunto de datos del tren utilizando el mejor conjunto de hiperparámetros.

# The usual attributes and methods are available:
tvs_model.bestModel.intercept
tvs_model.bestModel.coefficients
tvs_model.bestModel.summary.rootMeanSquaredError
tvs_model.bestModel.evaluate(test).r2


# ## Ajuste los hiperparámetros utilizando la validación cruzada k-fold


# Para conjuntos de datos pequeños o ruidosos, la validación cruzada de k-fold puede ser más apropiada.
# Usar la clase
# [CrossValidator](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator)
# para especificar  k-fold cross-validation:
from pyspark.ml.tuning import CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3, seed=54321)
%time cv_model = cv.fit(train)

# El resultado es una instancia de la clase
# [CrossValidatorModel](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidatorModel)
# :
type(cv_model)



# Los resultados de validación cruzada se almacenan en el atributo `avgMetrics`:
cv_model.avgMetrics
def plot_kfold_results(model):
  plt.plot(regParamList, model.avgMetrics)
  plt.title("Hyperparameter Tuning Results")
  plt.xlabel("Regularization Parameter")
  plt.ylabel("Validation Metric")
  plt.show()
plot_kfold_results(cv_model)

# El atributo `bestModel` contiene el modelo basado en el mejor conjunto de
# hiperparámetros. En este caso, es una instancia de la
# Clase `LinearRegressionModel`:
type(cv_model.bestModel)



# Calcular el rendimiento del rendimiento del mejor modelo en la prueba
# conjunto de datos:
cv_model.bestModel.evaluate(test).r2


# ## Ejercicios

# (1) Quizás nuestros parámetros de regularización son demasiado grandes. Vuelva a ejecutar el
# ajuste de hiperparámetros con parámetros de regularización [0.0, 0.002, 0.004, 0.006,
# 0.008, 0.01].


# ## Referencias

# [Spark Documentation - Model Selection and hyperparameter
# tuning](http://spark.apache.org/docs/latest/ml-tuning.html)

# [Spark Python API - pyspark.ml.tuning
# module](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.tuning)


# ## Para la sesion

spark.stop()

