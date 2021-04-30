import csv
import datetime
import json
import numpy as np
from pyspark import SparkContext
import sys

categories = {
    "452210": "big_box_grocers",
    "452311": "big_box_grocers",
    "445120": "convenience_stores",
    "722410": "drinking_places",
    "722511": "full_service_restaurants",
    "722513": "limited_service_restaurants",
    "446110": "pharmacies_and_drug_stores",
    "446191": "pharmacies_and_drug_stores",
    "311811": "snack_and_bakeries",
    "722515": "snack_and_bakeries",
    "445210": "specialty_food_stores",
    "445220": "specialty_food_stores",
    "445230": "specialty_food_stores",
    "445291": "specialty_food_stores",
    "445292": "specialty_food_stores",
    "445299": "specialty_food_stores",
    "445110": "supermarkets_except_convenience_stores"
}

category_names = tuple(set(categories.values()))
naics_codes = tuple(set(categories.keys()))
headers = tuple(('year', 'date', 'median', 'low', 'high'))


# extract categories with naics codes
def extract_categories(partId, records):
    if partId == 0:
        next(records)

    reader = csv.reader(records)
    for row in reader:
        if row[9] in naics_codes:
            # yield place id by looking at naics code
            yield row[1], row[9]


def extract_visits(partId, records):
    if partId == 0:
        next(records)
    reader = csv.reader(records)

    for row in reader:
        # yield place_id, start_date and visits_by_day
        yield row[1], (row[12], row[16])


def date_conversion(x):
    start_date = datetime.datetime.strptime(x[0], "%Y-%m-%d")
    # end_date = datetime.datetime.strptime(x[1][1][:10], "%Y-%m-%d")
    visits_by_day = json.loads(x[1])
    return [((start_date + datetime.timedelta(days=day)).date().isoformat(), [int(visit)])
            for day, visit in enumerate(visits_by_day[1:])]


def computations(x):
    year = x[0][:4]
    date = x[0][5:]
    median = np.median(x[1])
    stdev = np.std(x[1])
    low = max(0, int(median - stdev))
    high = max(0, int(median + stdev))
    return str(year), '2020-' + str(date), int(median), low, high

def join_csv(database):
    return ','.join([str(data) for data in database])


def main(sc):
    core_places = sc.textFile('hdfs:///data/share/bdm/core-places-nyc.csv')
    weekly_pattern = sc.textFile('hdfs:///data/share/bdm/weekly-patterns-nyc-2019-2020/*')

    for file in category_names:
        place_id = set(core_places \
                       # .map(lambda x: x.split(',')) \
                       .mapPartitionsWithIndex(extract_categories) \
                       .map(lambda x: x[0]) \
                       .collect())
        date_visits = weekly_pattern \
            .mapPartitionsWithIndex(extract_visits) \
            .filter(lambda x: x[0] in place_id) \
            .map(lambda x: (x[1][0][:10], x[1][1])) \
            .filter(lambda x: x[0][:4] != '2018') \
            .flatMap(date_conversion) \
            .reduceByKey(lambda x, y: x + y) \
            .map(computations) \
            .sortBy(lambda x: x[0]) \
            .map(join_csv())

        headers.union(date_visits).saveAsTextFile(output_prefix + '/' + file)


if __name__ == '__main__':
    sc = SparkContext()
    output_prefix = sys.argv[-1]
    main(sc)
