<<<<<<< HEAD
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

=======
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

>>>>>>> bdd2750e416964698f1ddbe1736dcfb1853f2963
    return timestump