

# ## 

# Explorar los datos


# ## Posibles flujos de trabajo con big data

# * Trabajar con todos los datos en el cluster
#   * Producir reportes eficazmente
#   * Limitarse a reportes tabulares 
#   * Requiere bastnate computo 
# * Trabjar con Work with a sample of the data in local memory
#   * Abrir un amplio rango de herramientas 
#   * Enables more rapid iterations
#   * Produces sampled results
# * Summarize on the cluster and visualize summarized data in local memory
#   * Produces accurate counts
#   * Allows for wide range of analysis and visualization tools


# ## Setup

# Import useful packages, modules, classes, and functions:
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# crear la sesion 
spark = SparkSession.builder.master("local").appName("explore").getOrCreate()

# Leer los datos :
rides_sdf = spark.read.parquet("/duocar/joined-all")

# crear una muestra aleatoria y mandarla a 
# un dataframe de Pandas:
rides_pdf = rides_sdf.sample(withReplacement=False, fraction=0.01, seed=12345).toPandas()

# **Nota:** se usará la notacion  `sdf` para  Spark DataFrames y
# `pdf` para  pandas DataFrames.


# ## Explorando una variable 



# ### Variable categorica

# Explorar el tipo de vehiculo, el cual es un ejemplo de
# variable categorica .

# Suar el metodo  `groupBy` method in Spark para crear una tabla de frecuencia de una sola variable:
summarized_sdf = rides_sdf.groupBy("service").count().orderBy("service")
summarized_sdf.show()

# pasar de Spark dataframe a Pandas dataframes:
summarized_pdf = summarized_sdf.toPandas()
summarized_pdf

# **Nota**: Recordar que estamos leyendo datos a memoria local 
# trate mos que los datos sumarizados sean pequeños.

# Especificar un orden en las categorias :
order = ["Car", "Noir", "Grand", "Elite"]

# Usar seaborn para mostrar los datos sumarizados :
sns.barplot(x="service", y="count", data=summarized_pdf, order=order)

# Usar seaborn para mostrar los datos de muestra :
sns.countplot(x="service", data=rides_pdf, order=order)

# **Note:** Los graficos muestran la misma informacion cualitativa 


# ### Explorar la variable continua 

# We can use the `describe` para mostrar las estadisticas básicas:
rides_sdf.describe("distance").show()

# agregar funciones para obtener informacion estadistica adicional curtosis (medicion valores atipicos):
# skewness (oblicuidad) distoricion de la curva a la campana simétrica o normal 
# skewness positivo la curva concentra más valores del lado derecho, media y mediana mayores a la moda
# skewness negativo la curva concentra más valores del lado izquierdo , media y mediana menores a la moda
rides_sdf.agg(skewness("distance"), kurtosis("distance")).show()

# We can use the `approxQuantile` method to compute approximate quantiles:
rides_sdf.approxQuantile("distance", probabilities=[0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], relativeError=0.1)

# **Note:** asignar  `relativeError = 0.0` para dejarlo exacto (posiblemente sea mas costoso)
# los cuantiles.

# Histograma con una muestra aleatoria del 1% 

sampled_pdf = rides_sdf \
  .select("distance") \
  .dropna() \
  .sample(withReplacement=False, fraction=0.01, seed=23456) \
  .toPandas()

# Usar seaborn para crear un histograma de datos muestra :
sns.distplot(sampled_pdf["distance"], kde=False)

# Usar seaborn para crear un histograma normalizado
# con un grafico superpuesto y el kernel de estimación de densidad

sns.distplot(sampled_pdf["distance"], kde=True, rug=True)

# Usar seaborn para crear un grafico de cajas el cual muestra
# informacion de los calculos obtenidos al usar el
# metodo `approxQuantile` method:
sns.boxplot(x="distance", data=sampled_pdf)

# # [Boxplot seaborn](https://seaborn.pydata.org/generated/seaborn.boxplot.html)

# ## Explorar par de variables

# ### Categoricas con categoricas 

# Ver la distribucion de genero de los que son estudiantes o no .

# crear una tabla de frecuencias de dos variables :
summarized_sdf = rides_sdf.groupBy("rider_student", "rider_gender").count().orderBy("rider_student", "rider_gender")
summarized_sdf.show()

# Convertir a un dataframe de pandas:
summarized_pdf = summarized_sdf.toPandas()
summarized_pdf

# Generar un grafico de barras :
hue_order = ["female", "male"]
sns.barplot(x="rider_student", y="count", hue="rider_gender", hue_order=hue_order, data=summarized_pdf)

# Reemplazar los valores nulos o missing por una etiqueta :
summarized_pdf = summarized_sdf.fillna("missing").toPandas()
hue_order = ["female", "male", "missing"]
sns.barplot(x="rider_student", y="count", hue="rider_gender", hue_order=hue_order, data=summarized_pdf)

# ### Categorico a continuo 

# Explorar la distribucion de la distancia de las carreras por la categoria de ser estudiante o no

# Producir reportes tabulares en Spark:
rides_sdf \
  .groupBy("rider_student") \
  .agg(count("distance"), mean("distance"), stddev("distance")) \
  .orderBy("rider_student") \
  .show()

# Alternativamente, producir visualizaciones sobre un ejemplo:
sampled_pdf = rides_sdf \
  .select("rider_student", "distance") \
  .sample(withReplacement=False, fraction=0.01, seed=34567) \
  .toPandas()

# Usar seaborn para producir un stripplot de la muestra de datos :
sns.stripplot(x="rider_student", y="distance", data=sampled_pdf, jitter=True)

# **Nota:** observe q los que no son estudiantes toman carreras largas .

# Ver otras formas de mostrar la data..


# ### Variable continua versus continua

# Usar metodos  `corr`, `covar_samp`, y `covar_pop` funciones agregadas para medir la relación 
# linear entre esas dos variables 
# the linear relationship between two variables:
rides_sdf.agg(corr("distance", "duration"),
              covar_samp("distance", "duration"),
              covar_pop("distance", "duration")).show()

# Usar la funcion  `jointplot` para producir un diagrama de dispersion 
# robusto sobre datos de muestra 
# *sampled* data:
sns.jointplot(x="distance", y="duration", data=rides_pdf)

# Superponer una regresion lineal en la grafica :
sns.jointplot(x="distance", y="duration", data=rides_pdf, kind="reg")

# superponer una regresion cuadratica  (order = 2):
sns.jointplot(x="distance", y="duration", data=rides_pdf, kind="reg", order=2)

# usar la funcion pairplot  para examinar pares una vez :
sampled_pdf = rides_sdf \
  .select("distance", "duration", hour("date_time")) \
  .dropna() \
  .sample(withReplacement=False, fraction=0.01, seed=45678) \
  .toPandas()
sns.pairplot(sampled_pdf)



# ## Referencias

# [The SciPy Stack](https://scipy.org/)

# [pandas](http://pandas.pydata.org/)

# [matplotlib](https://matplotlib.org/index.html)

# [seaborn](https://seaborn.pydata.org/index.html)

# [Bokeh](http://bokeh.pydata.org/en/latest/)

# [Plotly](https://plot.ly/)


# ## limpiar y cerrar sesion

spark.stop()
