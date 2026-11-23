# Example data

Three hourly datasets, ready to use. Each file has a `Date` column, the target
column and numeric input columns; gaps are left as NaN.

| File | Target | Period | Licence |
|---|---|---|---|
| `grws/Parquet/Doon_closest.parquet` | `Doon-QR-8677042_river_Flow` (m³/s) | 2010–2022 | GRCA Open Data Licence v2.0 / CC BY 4.0 |
| `grws/Parquet/Leggatt_closest.parquet` | `Leggatt-QR-8755042_river_Flow` (m³/s) | 2010–2022 | GRCA Open Data Licence v2.0 / CC BY 4.0 |
| `air_quality/Nongzhanguan.parquet` | `PM2.5_Label` (µg/m³) | 2013–2017 | CC BY 4.0 |

## Grand River watershed streamflow (GRWS)

The two streamflow files are unchanged copies of two datasets of the
Grand River Watershed Streamflow (GRWS) Datasets:

> Kemp, Jason, 2026, "Grand River watershed streamflow",
> https://doi.org/10.5683/SP3/CH3T6Z, Borealis, V2.

Each file holds one river gauge (discharge, the target, and stage) and the
closest climate station. The full collection has 17 gauges; download it with
`python scripts/download_grws.py` (`--list` shows the files).

Column names are `<station>-<SHEF code>-<series id>_<category>`: `QR` river
discharge, `HG` river stage, `TW` water temperature, `TA` air temperature,
`PN` precipitation, `UD`/`US` wind direction/speed. Targets end in `_Flow`
(use `label_tag: _Flow`).

The GRWS Datasets Contains Information made available under Grand River
Conservation Authority’s Open Data Licence v2.0 which permits the data to be
copied, adapted, published and distributed. When reusing GRWS Datasets which
are GRCA-derived, users must comply with that licence and include the
attribution statement: “Contains Information made available under Grand River
Conservation Authority’s Open Data Licence v2.0.”

For the full GRCA licence text, see:
https://data.grandriver.ca/docs/GRCA%20Open%20Data%20Licence%20v2.pdf

## Beijing air quality (Nongzhanguan station)

> Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset]. UCI Machine
> Learning Repository. https://doi.org/10.24432/C5RK5G. CC BY 4.0.

Values are unchanged from the station file
`PRSA_Data_Nongzhanguan_20130301-20170228.csv`, except: `Date` is built from
the year/month/day/hour columns (dropped, with `No` and `station`), `PM2.5`
is renamed `PM2.5_Label` (the target), and the wind direction `wd` is
converted from compass points to degrees.
