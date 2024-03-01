# MDS_ML_Project

This repository is for the development of our ML group project for Machine Learning (C24) course taken at the Hertie School in Spring 2024. Group members are: Daniyar Imanaliev, Johannes Müller, Lonny Chen, Minho Kang,

We take data from the Kaggle set ["Hourly energy generation and weather" in Spain](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather), which has pre-downloaded Data from the [ENTSO-E](https://uat-transparency.entsoe.eu/dashboard/show) platform together with weather data purchased by the original poster from [Open Weather API](#0).

The [energy dataset](https://github.com/MDS-Mountain-Club/MDS_ML_Project/blob/main/Data/energy_dataset.csv) contains the *variables*:

-   ***time***: date and hour of observation in UTC format from 2015-01-01 to 2018-12-31
-   *generation* of electricity in MW of following types: *biomass, fossil brown, coal/lignite, fossil coal-derived gas, fossil gas, fossil hard coal, fossil oil, fossil oil shale, fossil peat, geothermal, hydro pumped storage aggregated, hydro run-of-river and poundage, hydro water reservoir, marine, nuclear, other, other renewable, **solar**, waste, wind offshore, **wind onshore***
-   *forecast* for the next day in MW of: *solar day ahead, wind onshore day ahead, total load forecast*
-   *total load actual*: actual electricity demand in MW.
-   *price* for electricity in EUR/MWh: *day ahead* and *actual* market price .

The [weather dataset](https://github.com/MDS-Mountain-Club/MDS_ML_Project/blob/main/Data/weather_features.csv) contains *variables* for each of the five different *city_name*:

-   ***dt_iso***: Date and hour in ISO format,
-   ***wind_speed*** (m/s) and ***wind_deg*** (°)
-   ***humidity***: Relative humidity (%)
-   ***rain_1h***: Rainfall (last hour, mm) as well as ***rain_3h*** and *snow_3h* (last 3 hours, mm)
-   ***clouds_all***: Cloud cover (%)
-   *temp*: Temperature in Kelvin (incl. *min*/*max*)
-   pressure: Atmospheric pressure (hPa)
-   *weather_id*: Weather condition ID, *weather_main*: Main condition, *weather_description*: detailed condition, *weather_icon*: Icon code.
