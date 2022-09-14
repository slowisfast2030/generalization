import findspark
findspark.init()
from pyspark.sql import SparkSession

def get_session():
    spark = SparkSession.builder.master("local") \
        .appName("ai-golden-data" ) \
        .config("spark.driver.memory", "10g") \
        .config("hive.metastore.uris", "thrift://bigdata-0002:9807") \
        .enableHiveSupport() \
        .getOrCreate()
    spark.conf.set("spark.sql.crossJoin.enabled", "true")
    spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict")
    return spark

spark = get_session()

# 拉取订单的pair_created_time
time_df = spark.sql('''
                    select 
                        cv_id, jd_id, pair_created_time
                     from 
                         ai_prod.ads_ai_raw_label_a 
                     where 
                         ds=20220908
                 ''')

# sample
label_df = spark.sql('''
                    select 
                         cv_id, jd_id 
                    from 
                         ai_prod.ads_ai_raw_sample_cv_jd 
                    where 
                         ds=20220908
               ''')

df = label_df\
        .join(time_df, ['cv_id', 'jd_id'], 'inner')

# 保存数据
df.toPandas().to_csv('raw_label_order_time_20220908_spark.csv')
