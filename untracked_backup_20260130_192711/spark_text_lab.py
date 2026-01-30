import argparse
import re
from collections import Counter
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pymongo import MongoClient

STOPWORDS = {"is","a","the","to","with","and"}

WORD_RE = re.compile(r"\b[0-9a-zA-Z']+\b")


def tokenize(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def rdd_word_counts(spark, path):
    sc = spark.sparkContext
    lines = sc.textFile(path)
    words = lines.flatMap(lambda l: tokenize(l))
    counts = words.map(lambda w: (w,1)).reduceByKey(lambda a,b: a+b)
    return counts


def compare_stopwords(counts_rdd):
    # top 10 before
    total = counts_rdd.map(lambda x: x[1]).sum()
    top_before = counts_rdd.takeOrdered(10, key=lambda x: -x[1])
    # remove stopwords
    filtered = counts_rdd.filter(lambda x: x[0] not in STOPWORDS)
    top_after = filtered.takeOrdered(10, key=lambda x: -x[1])
    return total, top_before, top_after, filtered


def df_from_counts(spark, counts_rdd):
    df = counts_rdd.toDF(["word","count"]).withColumn("length", col("word").rlike(".").cast("int")*0+1)
    # compute accurate length
    df = df.rdd.map(lambda r: (r['word'], r['count'], len(r['word']))).toDF(["word","count","length"])
    total = df.groupBy().sum('count').collect()[0][0]
    df = df.withColumn('freq', col('count')/total)
    return df, total


def weighted_avg_word_length(df, total):
    # sum(length * count)/total
    s = df.rdd.map(lambda r: r['length']*r['count']).sum()
    return s/total


def ten_longest(df):
    return df.orderBy(col('length').desc()).limit(10).select('word','count','length').collect()


def filter_count_ge(df, n):
    total = df.groupBy().sum('count').collect()[0][0]
    filtered = df.filter(col('count') >= n)
    share = filtered.groupBy().sum('count').collect()[0][0] / total
    return filtered, share


def mongo_store_counts(counts_rdd, mongo_uri, db_name='spark_text_lab', coll_name='words'):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    coll = db[coll_name]
    # replace collection
    coll.drop()
    docs = counts_rdd.map(lambda wc: {'word': wc[0], 'count': int(wc[1]), 'length': len(wc[0])}).collect()
    if docs:
        coll.insert_many(docs)
        coll.create_index([('count', -1)])
    return coll


def query_mongo_by_length(mongo_uri, min_length=7, db_name='spark_text_lab', coll_name='words'):
    client = MongoClient(mongo_uri)
    coll = client[db_name][coll_name]
    return list(coll.find({'length': {'$gte': min_length}}).sort('count', -1))


def top_bigrams(spark, path, top_k=20):
    sc = spark.sparkContext
    lines = sc.textFile(path)
    tokens = lines.map(lambda l: tokenize(l)).filter(lambda t: t)
    bigrams = tokens.flatMap(lambda ws: [ (ws[i]+" "+ws[i+1],1) for i in range(len(ws)-1)])
    bigram_counts = bigrams.reduceByKey(lambda a,b: a+b)
    return bigram_counts.takeOrdered(top_k, key=lambda x:-x[1])


def compute_tfidf(spark, path):
    from pyspark.sql.functions import monotonically_increasing_id
    from pyspark.ml.feature import Tokenizer, HashingTF, IDF
    docs = spark.read.text(path).toDF('text')
    docs = docs.withColumn('id', monotonically_increasing_id())
    tokenizer = Tokenizer(inputCol='text', outputCol='words')
    wordsData = tokenizer.transform(docs)
    hashingTF = HashingTF(inputCol='words', outputCol='rawFeatures', numFeatures=1<<12)
    featurizedData = hashingTF.transform(wordsData)
    idf = IDF(inputCol='rawFeatures', outputCol='features')
    idfModel = idf.fit(featurizedData)
    res = idfModel.transform(featurizedData)
    return res.select('id','words','features')


def per_file_and_global_counts(spark, folder):
    sc = spark.sparkContext
    # wholeTextFiles returns (path, content)
    files = sc.wholeTextFiles(folder)
    per_file = files.mapValues(lambda text: Counter(tokenize(text))).map(lambda kv: (kv[0].split('/')[-1], dict(kv[1])))
    # global
    global_counts = files.flatMap(lambda kv: tokenize(kv[1])).map(lambda w: (w,1)).reduceByKey(lambda a,b: a+b)
    # top 20 keywords per file
    top_per_file = files.mapValues(lambda text: Counter(tokenize(text)).most_common(20)).collect()
    return per_file.collect(), global_counts


def store_global_and_perfile_mongo(per_file, global_counts_rdd, mongo_uri, db_name='spark_text_lab'):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    db.global_wordcount.drop()
    db.per_file_wordcount.drop()
    # insert global
    docs = global_counts_rdd.map(lambda wc: {'word': wc[0], 'count': int(wc[1])}).collect()
    if docs:
        db.global_wordcount.insert_many(docs)
        db.global_wordcount.create_index([('count', -1)])
    # per-file
    per_docs = []
    for fname, d in per_file:
        for w,c in d.items():
            per_docs.append({'file': fname, 'word': w, 'count': int(c)})
    if per_docs:
        db.per_file_wordcount.insert_many(per_docs)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='sample_corpus.txt')
    parser.add_argument('--mongo-uri', default=None)
    parser.add_argument('--folder', default=None, help='folder for exercise 6')
    args = parser.parse_args()

    spark = SparkSession.builder.master('local[*]').appName('SparkTextLab').getOrCreate()

    counts = rdd_word_counts(spark, args.corpus)
    total, top_before, top_after, filtered = compare_stopwords(counts)
    print('Top 10 before stopword removal:', top_before)
    print('Top 10 after stopword removal:', top_after)

    df, total_count = df_from_counts(spark, counts)
    print('Weighted avg word length:', weighted_avg_word_length(df, total_count))
    print('10 longest words:', ten_longest(df))
    filtered_df, share = filter_count_ge(df, 2)
    print('Share of total frequency for count>=2:', share)

    if args.mongo_uri:
        coll = mongo_store_counts(counts, args.mongo_uri)
        print('Inserted to MongoDB collection:', coll.full_name)
        long_words = query_mongo_by_length(args.mongo_uri, 7)
        print('Mongo query length>=7:', long_words[:20])

    print('Top bigrams:', top_bigrams(spark, args.corpus, 20))

    tfidf = compute_tfidf(spark, args.corpus)
    print('TF-IDF computed; sample rows:')
    for r in tfidf.take(5):
        print(r)

    if args.folder and args.mongo_uri:
        per_file, global_counts = per_file_and_global_counts(spark, args.folder)
        store_global_and_perfile_mongo(per_file, global_counts, args.mongo_uri)
        print('Per-file and global counts stored in MongoDB')

    spark.stop()

if __name__ == '__main__':
    main()
