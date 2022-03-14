import calendar
import datetime

import pandas as pd


MAX_DAY = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def replace_year_in_date(date: datetime.datetime, year: int):
    if date.month == 2 and date.day == 29 and not calendar.isleap(year):
        date -= pd.Timedelta(1, "d")
    return date.replace(year=year)


def replace_month_in_date(date: datetime.datetime, month: int):
    if date.day > MAX_DAY[month]:
        date -= pd.Timedelta(date.day - MAX_DAY[month], "d")
        if month == 2 and date.day == 28 and calendar.isleap(date.year):
            date += pd.Timedelta(1, "d")
    return date.replace(month=month)


def replace_dayofweek_in_date(date: datetime.datetime, dayofweek: int):
    date += pd.Timedelta(dayofweek - date.weekday(), "d")
    return date