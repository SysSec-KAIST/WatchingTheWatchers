import time
import shutil

from utils import cmd, close_all_app
from utils import match_image
from config import CORD_YOUTUBE_CENTER
from config import CORD_YOUTUBE_SETTING
from config import CORD_YOUTUBE_SECRET
from config import YOUTUBE_DUMMY
from config import RESOLUTION, RESOLUTION_PREFIX

import logging
import coloredlogs
import pprint as pp

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def change_resolution(resolution=None):
    if resolution is None:
        logger.error("No resolution is given")
        exit(1)

    # Click setup!
    target_img_fname = "screenshot_setup_%d.png"
    match_img_fname = "resolution_imgbase/edge_setup_icon.png"
    # match_img_fname = "resolution_imgbase/galaxy_note8_setup_icon.png"
    setup_cord = None
    cnt = 0
    while setup_cord is None:
        if cnt > 10:
            logger.error("[-] Error in change_resolution")
            return None

        cmd(
            "input tap %d %d" % CORD_YOUTUBE_CENTER, interval=0.3
        )  # click video first time
        cmd("input tap %d %d" % CORD_YOUTUBE_SETTING, interval=0.3)  # click video setting
        capture_screen(target_img_fname % cnt)
        setup_cord = match_image(target_img_fname % cnt, match_img_fname)
        cnt += 1

    cmd("input tap %d %d" % setup_cord)

    #    cmd("input keyevent KEYCODE_DPAD_DOWN", interval=0)
    #    cmd("input keyevent KEYCODE_DPAD_DOWN", interval=0)
    #    cmd("input keyevent KEYCODE_DPAD_DOWN", interval=0)
    #    cmd("input keyevent KEYCODE_ENTER", interval=1)
    #    #cmd("input tap 140 1350") # click target resolution

    # Click target resolution
    target_img_fname = "screenshot_resolution.png"
    match_img_fname = "resolution_imgbase/%s_%dp.png" % (RESOLUTION_PREFIX, resolution)
    capture_screen(target_img_fname)
    cord = match_image(target_img_fname, match_img_fname)
    if cord is None:
        logger.error("[-] There is no %dp resolution. Please check." % (resolution))
        return False
    else:
        cmd("input tap %d %d" % cord)  # click target resolution
        return True


def open_youtube(name, link, resolution=None, interval=60, log_fname=None):
    # Wait 5 second to load youtube app.
    cmd('am start -a android.intent.action.VIEW -d "%s"' % link, interval=5)
    sleep_time = 5

    # This takes about 5 seconds
    if resolution:
        status = change_resolution(resolution)
        if status is None:
            close_all_app()
            return False

        elif status == False:
            with open("no_resolution_list.txt", "a") as f:
                f.write(", ".join([name, link, str(resolution)]) + "\n")

        if name == "baby" and status == True:
            logger.warning("{}, {}, {}".format(name, link, resolution))
            send_mail("check this!", ", ".join([name, link, str(resolution)]) + "\n")
        sleep_time += 5
    else:
        if log_fname:
            save_ad_status(log_fname)
        else:
            logger.debug("Skip to save dummy")

    # Wait leftover interval before closing the video
    if interval > sleep_time:
        time.sleep(interval - sleep_time)

    close_all_app()
    return True


def open_youtube_app_only():
    # Wait 5 second to load youtube app.
    link = "https://youtube.com/watch?"
    cmd('am start -a android.intent.action.VIEW -d "%s"' % link, interval=5)
    sleep_time = 5


def enable_secret_mode():
    # open_youtube_app_only()
    time.sleep(2)

    target_img_fname = "screenshot_secret.png"
    match_img_fname = "resolution_imgbase/secret_mode_disabled.png"

    capture_screen(target_img_fname)
    cord = match_image(target_img_fname, match_img_fname)
    if cord is None:
        logger.warning("Secret mode is already enabled")
        return False
    else:
        cmd("input tap %d %d" % cord)  # click target resolution

    time.sleep(2)

    cmd("input tap %d %d" % CORD_YOUTUBE_SECRET)


def save_ad_status(name):
    img_fname = "ad_check.png"
    target_img_fname = "ad_check_%s.png" % name
    logger.debug("save file: %s" % target_img_fname)
    capture_screen(img_fname)
    shutil.copy(img_fname, "imgbase/%s.png" % (name))


def open_youtube_with_ad():
    flush_youtube()

    logger.debug("Open YT app only")
    open_youtube_app_only()

    # capture_screen()
    logger.debug("Enable secret mode")
    enable_secret_mode()
    logger.debug("Setting done")


def open_dummy_youtube():
    # This link is only used for resolution change.
    open_youtube("dummy", YOUTUBE_DUMMY, RESOLUTION, interval=10, log_fname=None)


def flush_youtube():
    cmd("pm clear com.google.android.youtube", interval=3)
    open_youtube("dummy", YOUTUBE_DUMMY, None, interval=10, log_fname=None)
    open_dummy_youtube()
