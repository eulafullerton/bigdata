from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("FastFood_ML_Project") \
    .enableHiveSupport() \
    .getOrCreate()

# Step 2: Load data from Hive
df = spark.sql("""
SELECT
    sales_in_millions,
    total_units,
    franchised_stores,
    company_stores,
    unit_change
FROM fastfood
WHERE sales_in_millions IS NOT NULL
AND total_units IS NOT NULL
AND franchised_stores IS NOT NULL
AND company_stores IS NOT NULL
AND unit_change IS NOT NULL
""")

# Step 3: Feature engineering
assembler = VectorAssembler(
    inputCols=[
        "total_units",
        "franchised_stores",
        "company_stores",
        "unit_change"
    ],
    outputCol="features"
)

data = assembler.transform(df).select("features", "sales_in_millions")

# Step 4: Split data
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

# Step 5: Train model
lr = LinearRegression(labelCol="sales_in_millions")
model = lr.fit(train_data)

# Step 6: Evaluate model
results = model.evaluate(test_data)

rmse = results.rootMeanSquaredError
r2 = results.r2

print("RMSE:", rmse)
print("R2:", r2)

# Step 7: Prepare HBase data
metrics = [
    ("metrics1", "cf:rmse", str(rmse)),
    ("metrics1", "cf:r2", str(r2))
]

# Step 8: Write to HBase
def write_to_hbase(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('fastfood_metrics')

    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})

    connection.close()

spark.sparkContext.parallelize(metrics).foreachPartition(write_to_hbase)

# Step 9: Stop Spark
spark.stop()