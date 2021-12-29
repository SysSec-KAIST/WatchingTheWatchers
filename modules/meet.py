import time

from utils import cmd, close_all_app

from config import CORD_MEET_1
from config import CORD_MEET_2

import logging
import coloredlogs
import pprint as pp

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

def open_meet(link, interval=60):
    cmd("am start -a android.intent.action.VIEW -d \"%s\"" % link, interval=5)
    sleep_time = 5

    cmd("input tap %d %d" % CORD_MEET_1, interval=0.5)
    cmd("input tap %d %d" % CORD_MEET_2, interval=0.5)

    # Wait leftover interval before closing the video
    if interval > sleep_time:
        time.sleep(interval-sleep_time)

    close_all_app()
    return True
