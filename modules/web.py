from utils import cmd, close_all_app

from config import CORD_WEB_1
from config import CORD_WEB_2
from config import CORD_WEB_3

import logging
import coloredlogs
import pprint as pp

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def open_web(link, interval=60):
    cmd("am start -a android.intent.action.VIEW -d \"https://%s\"" % link, interval=3)
    return True


def flush_web():
    cmd("pm clear com.android.chrome", interval=3)
    cmd("am start -a android.intent.action.VIEW -d \"https://naver.com\"", interval=3)

    cmd("input tap %d %d" % CORD_WEB_1, interval=0.5)
    cmd("input tap %d %d" % CORD_WEB_2, interval=1)
    cmd("input tap %d %d" % CORD_WEB_3, interval=1)
    close_all_app()
