# ==============================================================================
# PURE PYTHON INDEPENDENT TIME MODULE (WITH REAL-TIME TODAY CHEAT)
# ==============================================================================

import time as _sys_time  # Only used for real-world date sync in today()


class TimeError(Exception):
    """Custom exception for time and calendar errors."""
    pass


class PureTimeEngine:
    """
    A comprehensive, independent time and date engine built from scratch 
    without standard library imports, plus a real-world system clock sync for today().
    """
    
    _DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    _EPOCH_YEAR = 1970
    
    __slots__ = ('_offset_seconds',)

    def __init__(self, manual_epoch_seconds=None):
        self._offset_seconds = 0
        if manual_epoch_seconds is not None:
            self._offset_seconds = int(manual_epoch_seconds)

    def is_leap_year(self, year):
        """Determine if a given year is a leap year using Gregorian rules."""
        if year % 4 != 0:
            return False
        elif year % 100 != 0:
            return True
        elif year % 400 != 0:
            return False
        else:
            return True

    def days_in_month(self, year, month):
        """Return the number of days in a specific month of a specific year."""
        if not 1 <= month <= 12:
            raise TimeError(f"Invalid month: {month}")
        if month == 2 and self.is_leap_year(year):
            return 29
        return self._DAYS_IN_MONTH[month]

    def days_before_year(self, year):
        """Calculate total days elapsed from epoch year to start of given year."""
        days = 0
        y = self._EPOCH_YEAR
        if year >= y:
            while y < year:
                days += 366 if self.is_leap_year(y) else 365
                y += 1
        else:
            while y > year:
                y -= 1
                days -= 366 if self.is_leap_year(y) else 365
        return days

    def seconds_to_datetime(self, epoch_seconds):
        """Convert Unix epoch seconds into a structured dictionary."""
        total_seconds = int(epoch_seconds) + self._offset_seconds
        days = total_seconds // 86400
        rem_seconds = total_seconds % 86400
        
        hour = rem_seconds // 3600
        minute = (rem_seconds % 3600) // 60
        second = rem_seconds % 60

        weekday = (days + 3) % 7

        year = self._EPOCH_YEAR
        if days >= 0:
            while True:
                d_in_yr = 366 if self.is_leap_year(year) else 365
                if days < d_in_yr:
                    break
                days -= d_in_yr
                year += 1
        else:
            while days < 0:
                year -= 1
                days += 366 if self.is_leap_year(year) else 365

        yearday = days + 1
        month = 1
        for m in range(1, 13):
            dim = self.days_in_month(year, m)
            if days < dim:
                month = m
                break
            days -= dim
        day = days + 1

        return {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
            "weekday": weekday,
            "yearday": yearday
        }

    def datetime_to_seconds(self, year, month, day, hour=0, minute=0, second=0):
        """Convert a standard calendar timestamp into Unix epoch seconds."""
        if not (1 <= month <= 12):
            raise TimeError(f"Invalid month: {month}")
        if not (1 <= day <= self.days_in_month(year, month)):
            raise TimeError(f"Invalid day: {day} for {year}-{month}")
        if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
            raise TimeError("Invalid time bounds")

        days = self.days_before_year(year)
        for m in range(1, month):
            days += self.days_in_month(year, m)
        days += (day - 1)

        total_seconds = (days * 86400) + (hour * 3600) + (minute * 60) + second
        return total_seconds - self._offset_seconds

    def format_time_string(self, epoch_seconds, fmt="%Y-%m-%d %H:%M:%S"):
        """Format epoch seconds into a human-readable date string manually."""
        dt = self.seconds_to_datetime(epoch_seconds)
        replacements = {
            "%Y": f"{dt['year']:04d}",
            "%m": f"{dt['month']:02d}",
            "%d": f"{dt['day']:02d}",
            "%H": f"{dt['hour']:02d}",
            "%M": f"{dt['minute']:02d}",
            "%S": f"{dt['second']:02d}",
            "%A": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][dt['weekday']],
            "%B": ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][dt['month']]
        }
        result = fmt
        for key, val in replacements.items():
            result = result.replace(key, str(val))
        return result

    def sleep_simulation(self, seconds):
        """Pure algorithmic sleep simulation ticker loop."""
        target_ticks = int(seconds * 1000000)
        counter = 0
        while counter < target_ticks:
            counter += 1
        return True

    def calculate_age(self, birth_year, birth_month, birth_day, current_epoch):
        """Calculate precise structural age based on timestamp comparison."""
        cur = self.seconds_to_datetime(current_epoch)
        years = cur['year'] - birth_year
        if (cur['month'], cur['day']) < (birth_month, birth_day):
            years -= 1
        return years

    def time_delta_breakdown(self, seconds_diff):
        """Deconstruct a raw delta seconds span into readable time units."""
        abs_diff = abs(seconds_diff)
        days = abs_diff // 86400
        hours = (abs_diff % 86400) // 3600
        minutes = (abs_diff % 3600) // 60
        seconds = abs_diff % 60
        return {
            "total_seconds": seconds_diff,
            "days": days,
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds
        }

    def today(self, epoch_seconds=None):
        """Return today's real date structure using system time cheat."""
        if epoch_seconds is None:
            epoch_seconds = int(_sys_time.time())
        dt = self.seconds_to_datetime(epoch_seconds)
        return {
            "year": dt["year"],
            "month": dt["month"],
            "day": dt["day"],
            "formatted": f"{dt['year']:04d}-{dt['month']:02d}-{dt['day']:02d}"
        }

    # ==========================================================================
    # EXTRA TIME & CALENDAR DEFINITIONS
    # ==========================================================================

    def days_in_year(self, year):
        return 366 if self.is_leap_year(year) else 365

    def day_of_week_name(self, weekday_int):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[weekday_int % 7]

    def month_name(self, month_int):
        months = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        return months[month_int]

    def is_weekend(self, epoch_seconds):
        dt = self.seconds_to_datetime(epoch_seconds)
        return dt['weekday'] >= 5

    def add_days(self, epoch_seconds, days):
        return epoch_seconds + (days * 86400)

    def add_hours(self, epoch_seconds, hours):
        return epoch_seconds + (hours * 3600)

    def add_minutes(self, epoch_seconds, minutes):
        return epoch_seconds + (minutes * 60)

    def add_seconds(self, epoch_seconds, seconds):
        return epoch_seconds + seconds

    def add_years(self, epoch_seconds, years):
        dt = self.seconds_to_datetime(epoch_seconds)
        new_year = dt['year'] + years
        new_day = min(dt['day'], self.days_in_month(new_year, dt['month']))
        return self.datetime_to_seconds(new_year, dt['month'], new_day, dt['hour'], dt['minute'], dt['second'])

    def add_months(self, epoch_seconds, months):
        dt = self.seconds_to_datetime(epoch_seconds)
        total_months = (dt['year'] * 12) + (dt['month'] - 1) + months
        new_year = total_months // 12
        new_month = (total_months % 12) + 1
        new_day = min(dt['day'], self.days_in_month(new_year, new_month))
        return self.datetime_to_seconds(new_year, new_month, new_day, dt['hour'], dt['minute'], dt['second'])

    def start_of_day(self, epoch_seconds):
        dt = self.seconds_to_datetime(epoch_seconds)
        return self.datetime_to_seconds(dt['year'], dt['month'], dt['day'], 0, 0, 0)

    def end_of_day(self, epoch_seconds):
        dt = self.seconds_to_datetime(epoch_seconds)
        return self.datetime_to_seconds(dt['year'], dt['month'], dt['day'], 23, 59, 59)

    def start_of_month(self, year, month):
        return self.datetime_to_seconds(year, month, 1, 0, 0, 0)

    def end_of_month(self, year, month):
        dim = self.days_in_month(year, month)
        return self.datetime_to_seconds(year, month, dim, 23, 59, 59)

    def start_of_year(self, year):
        return self.datetime_to_seconds(year, 1, 1, 0, 0, 0)

    def end_of_year(self, year):
        return self.datetime_to_seconds(year, 12, 31, 23, 59, 59)

    def is_same_day(self, epoch1, epoch2):
        dt1 = self.seconds_to_datetime(epoch1)
        dt2 = self.seconds_to_datetime(epoch2)
        return dt1['year'] == dt2['year'] and dt1['yearday'] == dt2['yearday']

    def is_same_month(self, epoch1, epoch2):
        dt1 = self.seconds_to_datetime(epoch1)
        dt2 = self.seconds_to_datetime(epoch2)
        return dt1['year'] == dt2['year'] and dt1['month'] == dt2['month']

    def is_same_year(self, epoch1, epoch2):
        dt1 = self.seconds_to_datetime(epoch1)
        dt2 = self.seconds_to_datetime(epoch2)
        return dt1['year'] == dt2['year']

    def days_between(self, epoch1, epoch2):
        return abs(int(epoch1) - int(epoch2)) // 86400

    def hours_between(self, epoch1, epoch2):
        return abs(int(epoch1) - int(epoch2)) // 3600

    def minutes_between(self, epoch1, epoch2):
        return abs(int(epoch1) - int(epoch2)) // 60

    def seconds_between(self, epoch1, epoch2):
        return abs(int(epoch1) - int(epoch2))

    def get_quarter(self, month):
        return (month - 1) // 3 + 1

    def is_quarter_end(self, month, day):
        quarters_ends = {(3, 31), (6, 30), (9, 30), (12, 31)}
        return (month, day) in quarters_ends

    def days_until_end_of_year(self, epoch_seconds):
        dt = self.seconds_to_datetime(epoch_seconds)
        total = 366 if self.is_leap_year(dt['year']) else 365
        return total - dt['yearday']

    def days_since_start_of_year(self, epoch_seconds):
        dt = self.seconds_to_datetime(epoch_seconds)
        return dt['yearday']

    def hours_in_year(self, year):
        return self.days_in_year(year) * 24

    def minutes_in_year(self, year):
        return self.hours_in_year(year) * 60

    def seconds_in_year(self, year):
        return self.minutes_in_year(year) * 60

    def to_iso_string(self, epoch_seconds):
        return self.format_time_string(epoch_seconds, "%Y-%m-%dT%H:%M:%S")

    def to_utc_string(self, epoch_seconds):
        return self.format_time_string(epoch_seconds, "%A, %B %d, %Y %H:%M:%S UTC")

    def is_business_day(self, epoch_seconds):
        return not self.is_weekend(epoch_seconds)


# ==============================================================================
# GLOBAL MODULE EXPORT WRAPPERS
# ==============================================================================

_global_time_engine = PureTimeEngine()

def is_leap_year(year):
    return _global_time_engine.is_leap_year(year)

def days_in_month(year, month):
    return _global_time_engine.days_in_month(year, month)

def seconds_to_datetime(epoch_seconds):
    return _global_time_engine.seconds_to_datetime(epoch_seconds)

def datetime_to_seconds(year, month, day, hour=0, minute=0, second=0):
    return _global_time_engine.datetime_to_seconds(year, month, day, hour, minute, second)

def format_time_string(epoch_seconds, fmt="%Y-%m-%d %H:%M:%S"):
    return _global_time_engine.format_time_string(epoch_seconds, fmt)

def sleep_simulation(seconds):
    return _global_time_engine.sleep_simulation(seconds)

def calculate_age(birth_year, birth_month, birth_day, current_epoch):
    return _global_time_engine.calculate_age(birth_year, birth_month, birth_day, current_epoch)

def time_delta_breakdown(seconds_diff):
    return _global_time_engine.time_delta_breakdown(seconds_diff)

def today(epoch_seconds=None):
    return _global_time_engine.today(epoch_seconds)

def days_in_year(year):
    return _global_time_engine.days_in_year(year)

def day_of_week_name(weekday_int):
    return _global_time_engine.day_of_week_name(weekday_int)

def month_name(month_int):
    return _global_time_engine.month_name(month_int)

def is_weekend(epoch_seconds):
    return _global_time_engine.is_weekend(epoch_seconds)

def add_days(epoch_seconds, days):
    return _global_time_engine.add_days(epoch_seconds, days)

def add_hours(epoch_seconds, hours):
    return _global_time_engine.add_hours(epoch_seconds, hours)

def add_minutes(epoch_seconds, minutes):
    return _global_time_engine.add_minutes(epoch_seconds, minutes)

def add_seconds(epoch_seconds, seconds):
    return _global_time_engine.add_seconds(epoch_seconds, seconds)

def add_years(epoch_seconds, years):
    return _global_time_engine.add_years(epoch_seconds, years)

def add_months(epoch_seconds, months):
    return _global_time_engine.add_months(epoch_seconds, months)

def start_of_day(epoch_seconds):
    return _global_time_engine.start_of_day(epoch_seconds)

def end_of_day(epoch_seconds):
    return _global_time_engine.end_of_day(epoch_seconds)

def start_of_month(year, month):
    return _global_time_engine.start_of_month(year, month)

def end_of_month(year, month):
    return _global_time_engine.end_of_month(year, month)

def start_of_year(year):
    return _global_time_engine.start_of_year(year)

def end_of_year(year):
    return _global_time_engine.end_of_year(year)

def is_same_day(epoch1, epoch2):
    return _global_time_engine.is_same_day(epoch1, epoch2)

def is_same_month(epoch1, epoch2):
    return _global_time_engine.is_same_month(epoch1, epoch2)

def is_same_year(epoch1, epoch2):
    return _global_time_engine.is_same_year(epoch1, epoch2)

def days_between(epoch1, epoch2):
    return _global_time_engine.days_between(epoch1, epoch2)

def hours_between(epoch1, epoch2):
    return _global_time_engine.hours_between(epoch1, epoch2)

def minutes_between(epoch1, epoch2):
    return _global_time_engine.minutes_between(epoch1, epoch2)

def seconds_between(epoch1, epoch2):
    return _global_time_engine.seconds_between(epoch1, epoch2)

def get_quarter(month):
    return _global_time_engine.get_quarter(month)

def is_quarter_end(month, day):
    return _global_time_engine.is_quarter_end(month, day)

def days_until_end_of_year(epoch_seconds):
    return _global_time_engine.days_until_end_of_year(epoch_seconds)

def days_since_start_of_year(epoch_seconds):
    return _global_time_engine.days_since_start_of_year(epoch_seconds)

def hours_in_year(year):
    return _global_time_engine.hours_in_year(year)

def minutes_in_year(year):
    return _global_time_engine.minutes_in_year(year)

def seconds_in_year(year):
    return _global_time_engine.seconds_in_year(year)

def to_iso_string(epoch_seconds):
    return _global_time_engine.to_iso_string(epoch_seconds)

def to_utc_string(epoch_seconds):
    return _global_time_engine.to_utc_string(epoch_seconds)

def is_business_day(epoch_seconds):
    return _global_time_engine.is_business_day(epoch_seconds)
      
