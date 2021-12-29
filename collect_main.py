import time
import random

from utils import check_adb
from utils import close_all_app
from utils import push_clear
from utils import toggle_airplane
from utils import start_airscope_pcell, stop_airscope_pcell, kill_airscope_pcell
from utils import start_log, end_log, send_log

from modules.youtube import open_youtube, flush_youtube, open_youtube_with_ad
from modules.web import open_web, flush_web
from modules.netflix import open_netflix, flush_netflix
from modules.meet import open_meet
from modules.amazon import open_amazon, flush_amazon

from config import RESOLUTION, LOGGING_TIME, START_IDX, TRIAL

import logging
import coloredlogs
import pprint as pp

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def start_test(opts, idx, name, link):
    if opts.youtube_ad:
        open_youtube_with_ad()

    kill_airscope_pcell()

    logger.info("===============================")
    logger.info("{}: {}, {}".format(idx, name, link))
    logger.info("===============================")
    if opts.web:
        name = "WEB"
    elif opts.meet:
        name = "MEET"
    log_fname = "%s_%03d" % (name, idx)

    logger.debug("\n\nToggle airplane")
    toggle_airplane()

    logger.debug("\n\nStart airscope")
    start_airscope_pcell(log_fname)
    logger.debug("\n\nWait for booting airscope (7sec)")
    time.sleep(4)

    logger.debug("Start dm log")
    start_log(log_fname)
    time.sleep(3)

    # TODO: make class hierarchy to make below abstract
    if opts.youtube:
        logger.debug("Open Youtube link")
        status = open_youtube(name, link, RESOLUTION, interval=LOGGING_TIME, log_fname=log_fname)
    elif opts.youtube_ad:
        logger.debug("Open Youtube link")
        status = open_youtube(name, link, None, interval=LOGGING_TIME, log_fname=log_fname)
    elif opts.netflix:
        status = open_netflix(link, interval=LOGGING_TIME)
    elif opts.web:
        status = open_web(link, interval=5)
    elif opts.meet:
        status = open_meet(link, interval=LOGGING_TIME)
    elif opts.amazon:
        status = open_amazon(link, interval=LOGGING_TIME)

    logger.debug("Stop airscope")
    stop_airscope_ca(log_fname)

    logger.debug("End log")
    end_log(log_fname)

    logger.debug("Send log")
    send_log(log_fname)

    return status


def main(opts):
    orig_links = get_links(opts.links)
    logger.debug(pp.pformat(orig_links))

    check_adb()
    close_all_app()

    # move to optis screen
    logger.info("It starts after 5 seconds. "
                "Please move the mouse cursor to the Optis-S screen")
    time.sleep(5)
    push_clear()

    for idx in range(START_IDX, START_IDX + TRIAL):
        if opts.youtube:
            # first clear youtube cache and setup default resolution
            flush_youtube()
        elif opts.netflix:
            # first clear netflix cache and login with default account
            flush_netflix()
        elif opts.web:
            flush_web()
        elif opts.amazon:
        # first clear amazon cache
            flush_amazon()

        links = orig_links
        if opts.web:
            links = random.shuffle(orig_links)

        for name, link in links:
            name = name.strip()
            link = link.strip()
            if opts.netflix or opts.amazon:
                link = link.replace('title', 'watch')

            cnt = 1
            status = start_test(opts, idx, name, link)
            if opts.youtube:
                while status != True:
                    cnt += 1
                    send_mail("Error in testing ()", "%d-th trial, %d, %s, %s, %d" % (cnt, idx, name, link, RESOLUTION))
                    logger.warning("[+] Restarting ... %d th trial" % cnt)
                    flush_youtube()
                    status = start_test(opts, idx, name, link)


if __name__ == "__main__":
    import sys
    from optparse import OptionParser

    op = OptionParser()
    op.add_option("--youtube", action="store_true", dest="youtube")
    op.add_option("--youtube_ad", action="store_true", dest="youtube_ad")
    op.add_option("--netflix", action="store_true", dest="netflix")
    op.add_option("--web", action="store_true", dest="web")
    op.add_option("--meet", action="store_true", dest="meet")
    op.add_option(
        "--links",
        action="store",
        type=str,
        dest="List of links",
        help="A file containing target links."
    )
    op.add_option("--debug", action="store_true", dest="debug")

    (opts, args) = op.parse_args()
    if not any([opts.youtube, opts.youtube_ad, opts.netflix, opts.web, opts.meet]):
        print("You should give at least one target.")
        op.print_help()
        exit(1)

    if not opts.links:
        print("You should give links.")
        op.print_help()
        exit(1)

    main(opts)
