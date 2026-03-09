import time

import happybase
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

HIVE_DB = "final_project"
HIVE_TABLE = "noshowappointments"
HBASE_TABLE = "project_metrics"
TARGET_COL = "no_show"
FEATURE_COLS = [
    "age",
    "scholarship",
    "hipertension",
    "diabetes",
    "alcoholism",
    "handcap",
    "sms_received",
]

spark = SparkSession.builder \
    .appName("FinalProjectML_to_HBase") \
    .enableHiveSupport() \
    .getOrCreate()

print("Spark App ID:", spark.sparkContext.applicationId)

full_table = f"{HIVE_DB}.{HIVE_TABLE}"
df = spark.sql(
    f"""
    SELECT age, scholarship, hipertension, diabetes, alcoholism, handcap, sms_received, no_show
    FROM {full_table}
    """
).dropna()

df = df.withColumn(
    "label",
    when(col(TARGET_COL) == "Yes", 1.0).otherwise(0.0)
)

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features", handleInvalid="skip")
ml_df = assembler.transform(df).select("features", "label")

train, test = ml_df.randomSplit([0.8, 0.2], seed=42)
model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
fitted = model.fit(train)
pred = fitted.transform(test)

acc_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy",
)
f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1",
)

metrics = {
    "accuracy": float(acc_eval.evaluate(pred)),
    "f1": float(f1_eval.evaluate(pred)),
}

print("Model metrics:", metrics)

# Write metrics to HBase
run_key = f"run_{int(time.time())}"
rows = [(run_key, f"cf:{k}", str(v)) for k, v in metrics.items()]

def write_partition(partition):
    conn = happybase.Connection("master")
    conn.open()
    table = conn.table(HBASE_TABLE)
    for row_key, col_name, val in partition:
        table.put(row_key, {col_name: val})
    conn.close()

spark.sparkContext.parallelize(rows, 1).foreachPartition(write_partition)

spark.stop()