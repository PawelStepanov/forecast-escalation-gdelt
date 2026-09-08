-- Primary source: GDELT Events, partitioned public BigQuery table.
-- Country: Israel. ActionGeo_CountryCode uses GDELT/FIPS-style code `IS`.
-- Weekly frequency: Monday-Sunday.
-- Target: count of CAMEO root 18 (Assault) and 19 (Fight).

SELECT
  DATE_TRUNC(
    PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)),
    WEEK(MONDAY)
  ) AS week_start,

  COUNTIF(EventRootCode IN ('18', '19')) AS conflict_count,
  COUNT(*) AS total_events_country,

  AVG(IF(EventRootCode IN ('18', '19'), GoldsteinScale, NULL)) AS avg_goldstein,
  STDDEV_SAMP(IF(EventRootCode IN ('18', '19'), GoldsteinScale, NULL)) AS goldstein_std,
  AVG(IF(EventRootCode IN ('18', '19'), AvgTone, NULL)) AS avg_tone,

  SUM(IF(EventRootCode IN ('18', '19'), COALESCE(NumMentions, 0), 0)) AS conflict_mentions,
  SUM(COALESCE(NumMentions, 0)) AS total_mentions_country

FROM `gdelt-bq.gdeltv2.events_partitioned`
WHERE
  _PARTITIONTIME >= TIMESTAMP('2022-01-03')
  AND _PARTITIONTIME < TIMESTAMP('2026-08-31')
  AND ActionGeo_CountryCode = 'IS'
GROUP BY week_start
ORDER BY week_start;
