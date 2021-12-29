import time

from utils import cmd, close_all_app, close_all_app_vert

import logging
import coloredlogs
import pprint as pp

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def flush_amazon():
    cmd("pm clear com.amazon.avod.thirdpartyclient", interval=3)


def open_amazon(link, interval=60):
    # Actual loading the video.
    cmd("am start -a android.intent.action.VIEW -d \"%s\"" % link, interval)
    # move the playtime to the end of the movie.

    # cmd("input tap 1600 850", interval=0.5)
    # cmd("input swipe 1600 850 1900 850 500", interval=5)
    close_all_app_vert()
    close_all_app()
