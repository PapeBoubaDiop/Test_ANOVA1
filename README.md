# Test_ANOVA1


hdfs dfs -mkdir /projetBigData
hdfs dfs -mkdir /projetBigData/data
hdfs dfs -mkdir /projetBigData/data/by_year
curl https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2025.csv.gz -O | hdfs dfs -put - /projetBigData/data/by_year/2025.csv.gz

hdfs dfs -ls /projetBigData/data/by_year/2025.csv.gz
hdfs dfs -text /projetBigData/data/by_year/2025.csv.gz | head

