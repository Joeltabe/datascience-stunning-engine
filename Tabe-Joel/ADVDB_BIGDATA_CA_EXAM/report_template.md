# CA Exam Report — Advanced Database & Big Data

**Course:** Advanced Database and Big Data  
**Student Name:** [Your Name]  
**Matric No.:** [Your Matric Number]  
**Date:** February 3, 2026  

## 1. Environment
- OS / Machine: Google Colab (Linux-based)
- MongoDB version: MongoDB Atlas (Cloud)
- Spark version: PySpark 3.x (local mode)
- Hadoop/HDFS version (if available): Not locally available, used command-line examples
- How you started services (commands): Used MongoDB Atlas connection string, Spark initialized in Colab

## 2. MongoDB Tasks Summary
- Collections created: customers (from JSON lines), orders (from CSV), order_items (from CSV), products (from CSV)
- Indexes created: orders.customer_id, orders.order_ts, order_items.order_id, customers.country
- Key queries/aggregations (with outputs):
  - Top 5 customers in Cameroon by spending: Listed customer IDs with totals
  - Updated loyalty points for high spenders (>= $300): Modified X customers
  - Deleted old cancelled orders: Deleted Y orders
  - Category aggregation: Revenue and top 3 products per category (e.g., Electronics: $XXXX, top products listed)

## 3. PySpark Tasks Summary
- Data ingestion steps: Loaded CSV files (orders, order_items, products) and JSON lines (events) using Spark DataFrames
- Cleaning steps: Converted timestamps, added order_day/week, validated amounts with error calculation, identified suspicious orders
- Analytics outputs (top KPIs):
  - Top 10 products by revenue
  - Revenue by category per week (pivot table)
  - Repeat customer rate: XX.XX%
  - Conversion rate per device: Desktop XX.XX%, Mobile XX.XX%
  - Average events per session: X.XX
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
  1. Raw data stored in MongoDB collections
  2. Daily Spark job reads from MongoDB, computes KPIs
  3. Results written to HDFS as Parquet and back to MongoDB weekly_kpis collection
- Validation checks you applied (data quality): Check for nulls, revenue > 0, row counts match expected

## 6. Discussion
- Findings (insights): Identified top revenue categories, customer repeat rates, device conversion differences
- Challenges faced: Handling JSON lines parsing, timestamp conversions, aggregation pipelines
- Improvements / Next steps: Add search query tracking in events, implement real HDFS cluster, optimize Spark jobs with partitioning

## Appendix
- Screenshots / logs (optional): Notebook outputs included in submitted PDF/HTML

---

# Checklist before submission
- [ ] Notebook contains your answers and outputs
- [ ] Exported PDF/HTML included
- [ ] Report template completed
