CREATE EXTENSION postgis;

DROP TABLE if exists van_local_area_bounds;
CREATE TABLE van_local_area_bounds (
    "MAPID" varchar(6) PRIMARY KEY,
    "NAME" varchar(128),
    geom geometry
);

DROP TABLE if exists van_city_bounds;
CREATE TABLE van_city_bounds (
    "FID" integer PRIMARY KEY,
    geom geometry
);


DROP TABLE if exists boundaries;
CREATE TABLE boundaries (
    "FID" integer PRIMARY KEY,
    "Name" char(3),
    ""
    geom geometry
);




\COPY incits_38 FROM '../../data/csv/incits_38.csv' WITH (FORMAT csv);