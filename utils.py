import os
import glob
import time
import cv2
import numpy as np
import pyautogui

from config import CORD_CLOSE_APP, CORD_CLOSE_APP_VERT
from config import CORD_TOGGLE_AIRPLANE
from config import BTN_LOGGING
from config import BTN_EXPORT
from config import BTN_EXPORT_LOGNAME
from config import BTN_EXPORT_EXPORT
from config import BTN_EXPORT_OK
from config import BTN_CLEAR
from config import HOST_IP, HOST_PORT, USERNAME, PASSWD
from config import AIRSCOPE_BIN_PATH, AIRSCOPE_CONFIG_PATH, AIRSCOPE_LOG_DIR

# TODO: integrate scell implementation with RF_NUM
from config import RF_NUM

import logging
import coloredlogs
import pprint as pp

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

def cmd(cmd, interval=1):
    os.system("adb shell " + cmd)
    time.sleep(interval)


def check_adb():
    os.system("adb devices")
    time.sleep(3)


def close_all_app():
    cmd("input keyevent KEYCODE_APP_SWITCH", interval=1)
    cmd("input tap %d %d" % CORD_CLOSE_APP, interval=1)
    #cmd("input keyevent KEYCODE_DPAD_DOWN", interval=3)
    #cmd("input keyevent KEYCODE_ENTER", interval=3)
    #cmd("input keyevent KEYCODE_HOME")

def close_all_app_vert():
    cmd("input keyevent KEYCODE_APP_SWITCH", interval=1)
    cmd("input tap %d %d" % CORD_CLOSE_APP_VERT, interval=1)
    #cmd("input keyevent KEYCODE_DPAD_DOWN", interval=3)
    #cmd("input keyevent KEYCODE_ENTER", interval=3)
    #cmd("input keyevent KEYCODE_HOME")


def toggle_airplane(interval=10):
    cmd("am start -a android.settings.AIRPLANE_MODE_SETTINGS", 5)
    logger.debug('click to go airplane mode')
    # below enter_airplane is only used for Note 4
    #cmd("input tap %d %d" % (cord_enter_airplane), interval=1) # Go to airplane
    cmd("input tap %d %d" % (CORD_TOGGLE_AIRPLANE), interval) # detach
    logger.debug('click to come back from airplane mode')
    cmd("input tap %d %d" % (CORD_TOGGLE_AIRPLANE), interval=2) # reattach
    cmd("input keyevent KEYCODE_HOME")


def find_image(im, tpl):
    im = np.atleast_3d(im)
    tpl = np.atleast_3d(tpl)
    H, W, D = im.shape[:3]
    h, w = tpl.shape[:2]

    # Integral image and template sum per channel
    sat = im.cumsum(1).cumsum(0)
    tplsum = np.array([tpl[:, :, i].sum() for i in range(D)])

    # Calculate lookup table for all the possible windows
    iA, iB, iC, iD = sat[:-h, :-w], sat[:-h, w:], sat[h:, :-w], sat[h:, w:] 
    lookup = iD - iB - iC + iA
    # Possible matches
    possible_match = np.where(np.logical_and.reduce([lookup[..., i] == tplsum[i] for i in range(D)]))

    # Find exact match
    for y, x in zip(*possible_match):
        if np.all(im[y+1:y+h+1, x+1:x+w+1] == tpl):
            return (y+1, x+1)

    return None


def match_image(target_img_fname, match_img_fname):
    target_img = cv2.imread(target_img_fname)
    match_img = cv2.imread(match_img_fname)
    w, h = match_img.shape[:2]
    img = find_image(target_img, match_img)
    if img is None:
        return None
    else:
        #cv2.rectangle(target_img, img, (img[0]+h, img[1]+w), (0, 0, 255), 2)
        #cv2.imwrite('screenshot2.png', target_img)
        return img[::-1]


def capture_screen(fname='screenshot.png'):
    # Capture the screenshot and pull it to PC
    cmd("/system/bin/screencap -p /sdcard/%s" % fname)
    os.system("adb pull /sdcard/%s" % (fname))
    return fname


def get_links(fname):
    with open(fname, 'r') as f:
        links = f.read().splitlines()
    links = list(map(lambda x: x.split(','), links))
    return links


def click(pos, interval=1):
    pyautogui.click(pos)
    time.sleep(interval)


def write(s, interval=0.05):
    pyautogui.typewrite(s, interval=interval)


# Below is DM related code
def start_log(name):
    click(BTN_LOGGING)
    write(name + '\n')
    write('y')


def end_log(name):
    click(BTN_EXPORT)
    click(BTN_EXPORT_LOGNAME)
    write(name + '\n')
    click(BTN_EXPORT_EXPORT, interval=15)
    click(BTN_EXPORT_OK)

    click(BTN_LOGGING)
    write('y')

    push_clear()


def push_clear():
    click(BTN_CLEAR)
    write("y")


def send_mail(title, body):
    shell('python3 send_mail.py "%s" "%s"' % (title, body))


def send_log(log_fname):
    transport = paramiko.Transport((HOST_IP, PORT_ADDR))
    transport.connect(username=USERNAME, password=PASSWD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    logs = glob.glob('dmlogs/%s*' % (log_fname))
    for log in logs:
        sftp.put(log, os.path.join(AIRSCOPE_LOG_DIR, log.replace('dmlogs\\', ''))
        os.remove(log)
    sftp.close()
    transport.close()

def shell(cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST_IP, port=HOST_PORT, username=USERNAME, password=PASSWD)
    ssh.exec_command(cmd)
    time.sleep(3)
    ssh.close()

def kill_airscope_pcell():
    logger.debug('kill airscope @ PCELL')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST_IP, port=HOST_PORT, username=USERNAME, password=PASSWD)
    ssh.exec_command("ps aux | grep airscope | awk '{print $2}' | xargs sudo -S kill -9")
    time.sleep(3)
    ssh.close()

def move_logfile_pcell(name):
    logger.debug('sudo cp "/tmp/airscope_pcell.csv" "%s/airscope_pcell_%s.csv"' % (AIRSCOPE_LOG_DIR, name))
    logger.debug('sudo cp "/tmp/airscope_pcell.log" "%s/airscope_pcell_%s.log"' % (AIRSCOPE_LOG_DIR, name))
    logger.debug('sudo cp "/tmp/airscope_pcell.pcap" "%s/airscope_pcell_%s.pcap"' % (AIRSCOPE_LOG_DIR, name))

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST_IP, port=HOST_PORT, username=USERNAME, password=PASSWD)

    ssh.exec_command('sudo cp "/tmp/airscope_pcell.csv" "%s/airscope_pcell_%s.csv"' % (AIRSCOPE_LOG_DIR, name))
    ssh.exec_command('sudo cp "/tmp/airscope_pcell.log" "%s/airscope_pcell_%s.log"' % (AIRSCOPE_LOG_DIR, name))
    ssh.exec_command('sudo cp "/tmp/airscope_pcell.pcap" "%s/airscope_pcell_%s.pcap"' % (AIRSCOPE_LOG_DIR, name))
    time.sleep(5)
    ssh.close()


def stop_airscope_pcell(name):
    logger.debug('kill airscope and move log file')
    kill_airscope_pcell()
    logger.debug('Move logfiles')
    move_logfile_pcell(name)
    time.sleep(3)


def start_airscope_pcell(log_fname):
    logger.debug('start airscope pcell at {}:{}, {}'.format(HOST_IP, HOST_PORT, USERNAME))

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST_IP, port=HOST_PORT, username=USERNAME, password=PASSWD)
    sudo_cmd = 'sudo -S {0} "{1}" > {2}/{3}_stdout.out 2> {2}/{3}_stderr.out'.format(AIRSCOPE_BIN_PATH, AIRSCOPE_CONFIG_PATH, AIRSCOPE_LOG_DIR, log_fname)
    logger.debug(sudo_cmd)

    stdin, stdout, stderr = ssh.exec_command(sudo_cmd)
    stdin.write(PASSWD + "\n")
    stdin.flush()
    ssh.close()
