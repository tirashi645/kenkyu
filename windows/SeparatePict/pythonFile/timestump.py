def get_time():
    import datetime

    # 時間取得（タイムスタンプ）

    now = datetime.datetime.today()

    year = str(now.year)
    month = str(now.month)
    day = str(now.day)
    hour = str(now.hour)
    minute = str(now.minute)
    second = str(now.second)

    timestump = month + "_" + day + "_"  + hour + "_"  + minute + "_" + second

    return timestump