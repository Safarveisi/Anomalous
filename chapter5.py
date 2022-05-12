from typing import Tuple
from random import randint
from functools import partial
from pyspark.sql import Window
import pyspark.ml.feature as ft
import pyspark.sql.functions as fn
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.sql import types, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, OneHotEncoder, StringIndexer

# Create a Spark application
spark = SparkSession.builder\
        .appName('NetworkAttack')\
        .master('local[4]')\
        .getOrCreate()

# Read the comma-delimited file 
# You can get it from http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
dataWithoutHeader = spark.read. \
 option("inferSchema", "true"). \
 option("header", "false"). \
 csv("kddcup.data.gz")

# The csv we have just imported, does not have the names for the columns
# Specify the names here
data = dataWithoutHeader.toDF(
 "duration", "protocol_type", "service", "flag",
 "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
 "hot", "num_failed_logins", "logged_in", "num_compromised",
 "root_shell", "su_attempted", "num_root", "num_file_creations",
 "num_shells", "num_access_files", "num_outbound_cmds",
 "is_host_login", "is_guest_login", "count", "srv_count",
 "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
 "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
 "dst_host_count", "dst_host_srv_count",
 "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
 "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
 "dst_host_serror_rate", "dst_host_srv_serror_rate",
 "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
 "label")

print('Count of rows: {0}'.format(data.count()))

# The number of rows per each label (~cluster)
data.select("label").groupBy("label").count().orderBy(fn.desc("count")).show(25)

# Drop the categorical columns and cache the dataframe
numericOnly = data.drop('protocol_type', 'service', 'flag').cache()

# We are about to use an ML algorithm (k-means)
# We need to gather all columns except the target into one (assembling)
assembler = ft.VectorAssembler(
    inputCols=list(
        filter(
        lambda col: col != 'label',
        numericOnly.columns
        )
    ),
    outputCol='featureVector'
)

# Configure the k-means algorithm
kmeans = KMeans() \
    .setPredictionCol('cluster') \
    .setFeaturesCol('featureVector')
# Set the pipeline (assembler -> k-means)
pipeline = Pipeline().setStages([assembler, kmeans])
# Train the algorithm (last phase of the pipeline)
pipelineModel = pipeline.fit(numericOnly)
# Get the centers of clusters (k-means appeared at the last stage)
pipelineModel.stages[-1].clusterCenters()
# Create a new column in the dataframe (using the transform method of the trained algorithm) 
# which shows to which cluster each row belongs
withCluster = pipelineModel.transform(numericOnly)
# Check how well the clustering went
# Note that in some datasets, the label may not be available
# In this case, the following is not applicable
withCluster.select('cluster', 'label') \
    .groupBy('cluster', 'label').count() \
    .orderBy('cluster', fn.desc('count')) \
    .show(25)

# Use the silhouette criterion (https://en.wikipedia.org/wiki/Silhouette_(clustering))
# to see how good each row fits to its cluster.
evaluator = ClusteringEvaluator() \
    .setFeaturesCol('featureVector') \
    .setPredictionCol('cluster')
    
silhouette = evaluator.evaluate(withCluster)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# We can put everything together
# Change the maximum number of iterations and tolerance
# to avoide early-stopping (by increasing the MaxIter) and 
# decrease the minimum amount of cluster centroid movement (Tol)
def clusteringScoreV1(data: DataFrame, k: int) -> float:
    assembler = ft.VectorAssembler() \
        .setInputCols(list(
        filter(
        lambda col: col != 'label',
        data.columns
        )
    )) \
        .setOutputCol('featureVector')
    
    kmeans = KMeans() \
        .setSeed(randint(1, 10)) \
        .setK(k) \
        .setTol(1.0e-5) \
        .setMaxIter(40) \
        .setPredictionCol('cluster') \
        .setFeaturesCol('featureVector')
    
    pipeline = Pipeline().setStages(
        [assembler, kmeans]
    )
    
    pipelineModel = pipeline.fit(data)
    withCluster = pipelineModel.transform(data)
    evaluator = ClusteringEvaluator() \
        .setFeaturesCol('featureVector') \
        .setPredictionCol('cluster')
    silhouette = evaluator.evaluate(withCluster)
    
    return silhouette

# Try different number of clusters 
Ks = [2, 4, 6]

kSelect = list(map(partial(clusteringScoreV1, numericOnly), Ks))

# Put the standard scaler in the pipeline
def clusteringScoreV2(data: DataFrame, k: int) -> float:
    assembler = ft.VectorAssembler() \
        .setInputCols(list(
        filter(
        lambda col: col != 'label',
        data.columns
        )
    )) \
        .setOutputCol('featureVector')
    
    scaler = StandardScaler() \
        .setInputCol('featureVector') \
        .setOutputCol('scaledFeatureVector') \
        .setWithMean(False) \
        .setWithStd(True)
    
    kmeans = KMeans() \
        .setSeed(randint(1, 10)) \
        .setK(k) \
        .setTol(1.0e-5) \
        .setMaxIter(40) \
        .setPredictionCol('cluster') \
        .setFeaturesCol('scaledFeatureVector')
    
    pipeline = Pipeline().setStages(
        [
            assembler,
            scaler,
            kmeans
        ]
    )
    
    pipelineModel = pipeline.fit(data)
    withCluster = pipelineModel.transform(data)
    evaluator = ClusteringEvaluator() \
        .setFeaturesCol('scaledFeatureVector') \
        .setPredictionCol('cluster')
    silhouette = evaluator.evaluate(withCluster)
    
    return silhouette

# Try k = 2
clusteringScoreV2(numericOnly, 2)

# Include the initally excluded features (protocol_type, service, flag)
# For this, we need a one-hot-encoder transformer
def oneHotPipeline(inputCol: str) -> Tuple[Pipeline, str]:
    
    indexer = StringIndexer() \
        .setInputCol(inputCol) \
        .setOutputCol(inputCol + '_indexed')
    
    encoder = OneHotEncoder() \
        .setInputCol(inputCol + '_indexed') \
        .setOutputCol(inputCol + '_vec')
    
    pipeline = Pipeline().setStages([indexer, encoder])
    
    return (pipeline, inputCol + '_vec')

# Make a composite pipeline ([oneHotPipeline, Pipeline])
def clusteringScoreV3(data: DataFrame, k: int = 2, returnSilhouette: bool = True) -> float:
    
    protoTypeEncoder, protoTypeVecCol = oneHotPipeline("protocol_type")
    serviceEncoder, serviceVecCol = oneHotPipeline("service")
    flagEncoder, flagVecCol = oneHotPipeline("flag")
    
    assembleCols = list(
        set(data.columns) - \
        set(['label', 'protocol_type', 'service', 'flag'])
        ) + [protoTypeVecCol, serviceVecCol, flagVecCol]
    
    assembler = ft.VectorAssembler() \
        .setInputCols(assembleCols) \
        .setOutputCol('featureVector')
    
    scaler = StandardScaler() \
        .setInputCol('featureVector') \
        .setOutputCol('scaledFeatureVector') \
        .setWithMean(False) \
        .setWithStd(True)
    
    kmeans = KMeans() \
        .setSeed(randint(1, 10)) \
        .setK(k) \
        .setTol(1.0e-5) \
        .setMaxIter(40) \
        .setPredictionCol('cluster') \
        .setFeaturesCol('scaledFeatureVector')
    
    pipeline = Pipeline().setStages(
        [
            protoTypeEncoder,
            serviceEncoder,
            flagEncoder,
            assembler,
            scaler,
            kmeans
        ]
    )
    
    pipelineModel = pipeline.fit(data)
    
    if returnSilhouette:
        withCluster = pipelineModel.transform(data)
        evaluator = ClusteringEvaluator() \
            .setFeaturesCol('scaledFeatureVector') \
            .setPredictionCol('cluster')
        silhouette = evaluator.evaluate(withCluster)
        return silhouette
    else:
        trainingCost = pipelineModel.stages[-1].summary.trainingCost
        return trainingCost
        
clusteringScoreV3(data, k=2, returnSilhouette=False)

# Again, try different number of clusters 
Ks = [2, 4, 6]

data.cache()
kTrainingCost = list(map(partial(clusteringScoreV3, data, returnSilhouette=False), Ks))
data.unpersist()

# A helper function which only outputs the trained k-means
# Similar to the other functions above
def fitPipeline(data: DataFrame, k: int = 2) -> PipelineModel:
    
    protoTypeEncoder, protoTypeVecCol = oneHotPipeline("protocol_type")
    serviceEncoder, serviceVecCol = oneHotPipeline("service")
    flagEncoder, flagVecCol = oneHotPipeline("flag")
    
    assembleCols = list(
        set(data.columns) - \
        set(['label', 'protocol_type', 'service', 'flag'])
        ) + [protoTypeVecCol, serviceVecCol, flagVecCol]
    
    assembler = ft.VectorAssembler() \
        .setInputCols(assembleCols) \
        .setOutputCol('featureVector')
    
    scaler = StandardScaler() \
        .setInputCol('featureVector') \
        .setOutputCol('scaledFeatureVector') \
        .setWithMean(False) \
        .setWithStd(True)
    
    kmeans = KMeans() \
        .setSeed(randint(1, 10)) \
        .setK(k) \
        .setTol(1.0e-5) \
        .setMaxIter(40) \
        .setPredictionCol('cluster') \
        .setFeaturesCol('scaledFeatureVector')
    
    pipeline = Pipeline().setStages(
        [
            protoTypeEncoder,
            serviceEncoder,
            flagEncoder,
            assembler,
            scaler,
            kmeans
        ]
    )
    
    pipelineModel = pipeline.fit(data)
    
    return pipelineModel

# Try k = 60
pipelineModel = fitPipeline(data, 60)

clusterLabel = pipelineModel.transform(data).select('cluster', 'label')

w = Window.partitionBy('cluster')

# Pyspark implementation of the weighted cluster entropy using the target column (as explained in the book)
weightedEntropy = clusterLabel.groupBy('cluster', 'label').count() \
    .withColumn('total', fn.sum('count').over(w)) \
    .withColumn('p', fn.col('count') / fn.col('total')) \
    .groupBy('cluster').agg((fn.sum('count') * -fn.sum(fn.col('p') * fn.log2(fn.col('p')))).alias('weighted_entropy')) \
    .rdd.map(lambda row: row[1]).collect()

# Weighted cluster entropy for k = 60
sum(weightedEntropy) / data.count()