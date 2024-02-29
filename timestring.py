import datetime

def timestring():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def iso8601():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")