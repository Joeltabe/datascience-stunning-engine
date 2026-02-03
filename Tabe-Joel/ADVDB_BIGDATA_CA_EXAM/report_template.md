# CA Exam Report — Advanced Database & Big Data

**Course:** Advanced Database and Big Data  
**Student Name:** Tabe Joel Etengeneng
**Matric No.:** [Your Matric Number]  
**Date:** February 3, 2026  

## 1. Environment
- OS / Machine: Google Colab (Linux-based)
- MongoDB version: MongoDB Atlas (Cloud)
- Spark version: v4.0.1
- Hadoop/HDFS version (if available):
- How you started services (commands): Used MongoDB Atlas connection string, Spark initialized in Colab and binded to Ngrok
    WARNING:pyngrok.process.ngrok:t=2026-02-03T15:44:34+0000 lvl=warn msg="can't bind default web address, trying alternatives" obj=web addr=127.0.0.1:4040
    Spark UI port: 4040
    Spark UI public URL: NgrokTunnel: "https://heedfully-nonworking-kina.ngrok-free.dev" -> "http://localhost:4040"

## 2. MongoDB Tasks Summary
- Collections created: customers (from JSON lines), orders (12,000 docs), order_items (27,556 docs), products (250 docs)
- Indexes created: orders.customer_id, orders.order_ts, order_items.order_id, customers.country
- Key queries/aggregations (with outputs):
  - Top 5 customers in Cameroon by spending:
    - C00059: $2,267.80
    - C00599: $2,100.65
    - C00239: $1,878.28
    - C00588: $1,841.40
    - C00695: $1,835.77
  - Updated loyalty points for high spenders (>= $300): 798 customers modified
  - Deleted old cancelled orders: 97 orders deleted
  - Category aggregation (delivered orders revenue):
    - Home: $26,710.19 | Top: P0024 ($1,573.43), P0138 ($1,482.33), P0090 ($1,364.93)
    - Stationery: $25,858.06 | Top: P0192 ($1,454.09), P0173 ($1,265.78), P0240 ($1,215.36)
    - Health: $22,843.95 | Top: P0068 ($1,622.73), P0110 ($1,576.25), P0109 ($1,503.37)
    - Fashion: $22,083.76 | Top: P0056 ($1,420.91), P0094 ($1,408.38), P0150 ($1,104.54)
    - Books: $21,450.15 | Top: P0136 ($1,644.79), P0135 ($1,341.53), P0116 ($1,256.15)
    - Groceries: $20,053.41 | Top: P0156 ($1,280.20), P0025 ($1,243.76), P0134 ($1,241.11)
    - Electronics: $19,904.18 | Top: P0189 ($2,187.36), P0200 ($1,992.81), P0118 ($1,887.19)

## 3. PySpark Tasks Summary
- Data ingestion steps: Loaded CSV files (orders: 12,000 rows, order_items: 27,556 rows, products: 250 rows) and JSON lines (events: 22,000 rows) using Spark DataFrames
- Cleaning steps: Converted timestamps, added order_day/week columns, validated amounts with error calculation
- Data quality check: 0 suspicious orders out of 12,000 (amount_error > 0.05)
- Analytics outputs (top KPIs):
  - Top 10 products by revenue:
    1. P0189 (Electronics): $2,187.36
    2. P0200 (Electronics): $1,992.81
    3. P0118 (Electronics): $1,887.19
    4. P0136 (Books): $1,644.79
    5. P0068 (Health): $1,622.73
    6. P0110 (Health): $1,576.25
    7. P0024 (Home): $1,573.43
    8. P0109 (Health): $1,503.37
    9. P0138 (Home): $1,482.33
    10. P0190 (Electronics): $1,476.86
  - Revenue by category per week: Pivot table generated (weeks 1, 36-52 across 7 categories)
  - Repeat customer rate: 79.55%
  - Conversion rate per device:
    - Android: 11.21% (380 payments / 3,390 views)
    - Web: 10.92% (363 payments / 3,323 views)
    - iOS: 9.72% (330 payments / 3,396 views)
  - Average events per session: 1.0
- Any optimization used (partitioning, caching): Used local[*] master for parallel processing

## 4. Hadoop/HDFS Tasks Summary
- HDFS directory structure: /campusmart/raw/ for data files
- Commands used for upload/download:
  - mkdir -p /campusmart/raw/
  - put data/* /campusmart/raw/
  - ls /campusmart/raw/ and du -h
  - get /campusmart/raw/orders.csv ./
- Spark-on-Hadoop submission command (if applicable): spark-submit --master yarn --deploy-mode cluster with memory configs

## 5. Integration Pipeline (MongoDB ↔ Spark ↔ HDFS)
- Your proposed architecture (bullets)
  1. Raw data stored in MongoDB collections (customers, orders, order_items, products)
  2. Daily Spark job reads from MongoDB, computes weekly KPIs by category
  3. Results written to HDFS as Parquet and back to MongoDB weekly_kpis collection (126 records inserted)
- Validation checks you applied (data quality): Check for nulls in category/revenue, revenue > 0, row counts match expected

## 6. Discussion
- Findings (insights):
  - Home category has highest total revenue ($26,710.19)
  - Electronics products dominate top revenue (4 of top 10)
  - High repeat customer rate (79.55%) indicates good retention
  - Android has best conversion rate (11.21%), iOS lowest (9.72%)
  - No data quality issues found (0 suspicious orders)
- Challenges faced: Handling JSON lines parsing, timestamp conversions, multi-collection aggregation pipelines with $lookup
- Improvements / Next steps: Add search query tracking in events schema, implement real HDFS cluster, optimize Spark jobs with partitioning and caching

## Appendix
- Screenshots / logs (optional): Notebook outputs included in submitted PDF/HTML

---

# Checklist before submission
- [x] Notebook contains your answers and outputs
- [ ] Exported PDF/HTML included
- [x] Report template completed
