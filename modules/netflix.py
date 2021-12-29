import time

from utils import cmd, close_all_app
from config import CORD_NETFLIX_LOGIN
from config import CORD_NETFLIX_PLAYSTART
from config import CORD_NETFLIX_PLAYEND
from config import NETFLIX_SWIPE_DURATION
from config import NETFLIX_DUMMY
from config import RESOLUTION, RESOLUTION_PREFIX

import logging
import coloredlogs
import pprint as pp

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def login_netflix():
    # First start is used for logging in to Netflix after flushing the cache and user data.
    cmd("am start -a android.intent.action.VIEW \"%s\"" % NETFLIX_DUMMY, interval=10)

    # select account (seunhe).
    cmd("input tap %d %d" % CORD_NETFLIX_LOGIN, interval=5)

#    # click yes
#    cmd("input tap 520 1620", interval=5)

    close_all_app()


def open_netflix(link, interval=60):
    # Actual loading the video.
    cmd('am start -a android.intent.action.VIEW -d "%s"' % link, interval)
    # Show the playtime
    cmd("input tap " % CORD_NETFLIX_PLAYSTART, interval=0.5)
    # Move the playtime to the end of the movie.
    cmd(
        "input swipe %d %d %d %d %d"
        % (*CORD_NETFLIX_PLAYSTART, *CORD_NETFLIX_PLAYEND, NETFLIX_SWIPE_DURATION),
        interval=5,
    )
    close_all_app()


def flush_netflix():
    cmd("pm clear com.netflix.mediaclient", interval=3)
    login_netflix()
