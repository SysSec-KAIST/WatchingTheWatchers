import os
import glob
import time
import pyautogui
import paramiko
import cv2
import numpy as np
import pprint as pp

# ========================= CONFIGURATION (Video Logging) ==================================
START_IDX = 10010
TRIAL = 40
#RESOLUTION = 1080
RESOLUTION = 720
#RESOLUTION = 480


#LOGGING_TIME = 200
LOGGING_TIME = 300
#VIDEO_LINK = 'youtube_links_u_skt1080.txt'
#VIDEO_LINK = 'youtube_links_despacito.txt'
#VIDEO_LINK = 'youtube_links_u.txt'
VIDEO_LINK = 'netflix_links.txt'
#

RF_NUM = 1
# ==========================================================================

# ========================= CONFIGURATION (Server) ==================================
# airscope binary and conf file location
#AIRSCOPE_PATH_PCELL = '/home/dkay/airscope_2018-05-02/airscope'
#CONFIG_PATH_PCELL = '/home/dkay/airscope_2018-05-02/conf/airscope_pcell.conf'

#AIRSCOPE_PATH_SCELL = '/home/dkay/airscope_2018-05-02/airscope'
#CONFIG_PATH_SCELL = '/home/dkay/airscope_2018-05-02/conf/airscope_scell.conf'
# ==========================================================================

# ==========================================================================
#LOGPATH = '/home/data/logs/KT_log'
#LOGPATH = '/home/data/logs/eval'
#LOGPATH = '/home/data/logs/SKT_10Mhz'
#LOGPATH = '/home/data/logs/KT_20Mhz_CA'
#LOGDIR = 'KT_20Mhz_CA_720'
#LOGDIR = 'KT_20Mhz_CA_1080'
#LOGDIR = 'KT_20Mhz_1080_NO_CA'
#LOGDIR = 'SKT_10Mhz_1080_NO_CA'
#LOGDIR = 'KT_10Mhz_B8_1080_NO_CA'
#LOGDIR = 'SKT_20Mhz_B3_720_NO_CA'
#LOGDIR = 'KT_20Mhz_B3_1080_NO_CA'
#LOGDIR = 'LG_20Mhz_B7_1080_NO_CA_2'
#LOGDIR = 'LG_20Mhz_B7_720_NO_CA'
#LOGDIR = 'LG_20Mhz_B7_480_NO_CA'
#LOGDIR = 'SKT_20Mhz_B3_1080_NO_CA_2'
#LOGDIR = 'KT_20Mhz_B3_1080_NO_CA_REAL'
#LOGDIR = 'KT_20Mhz_B3_B1_1080_CA'
#LOGDIR = 'KT_20Mhz_B3_B1_1080_3CA'
#LOGDIR = 'KT_20Mhz_B3_720_NO_CA_NEW'
#LOGDIR = 'KT_CA_B3_B1_1080_2CELL'
#LOGDIR = 'KT_CA_B3_B1_720_2CELL'
#LOGDIR = 'KT_20Mhz_B3_1080_NO_CA_EDGE2'

#LOGDIR = 'LG_20Mhz_B7_1080_NO_CA_EDGE'
#
#LOGDIR = 'KT_10Mhz_B8_1080_NO_CA_EDGE'
#LOGDIR = 'SKT_20Mhz_B3_1080_NO_CA_EDGE'
#LOGDIR = 'KT_10Mhz_B8_1080_NO_CA_EDGE_NEW_AS'
#LOGDIR = 'SKT_20Mhz_B3_1080_NO_CA_EDGE_NEW_AS'
#LOGDIR = 'KT_10Mhz_B8_1080_NO_CA_EDGE_OLD_AS_COMP'
#LOGDIR = "LG_10Mhz_B5_480_NO_CA_NOTE5"
LOGDIR = "LG_10Mhz_B5_netflix3"
#LOGDIR = "KT_10Mhz_B8_480_NO_CA_NOTE5"
#LOGDIR = "LG_10Mhz_B5_1080_NO_CA_NOTE5_new2"
#



# ==========================================================================
# PC addr, port
# ==========================================================================
EKPC_HOST = '143.248.230.232'
EKPC_PORT = 22
AIRSCOPE_PATH_EKPC = '/home/dkay/airscope_2018-05-02/airscope'
CONFIG_PATH_EKPC = '/home/dkay/airscope_2018-05-02/conf/airscope_scell.conf'
USERNAME_EKPC = 'dkay'
with open("passwd_dkay.txt", "r") as f:
    PASSWD_EKPC = f.read()
LOGPATH_EKPC = '/home/data/logs/%s' % (LOGDIR)

X300_HOST = '143.248.231.97'
X300_PORT = 22
#AIRSCOPE_PATH_X300 = '/home/hoops/airscope_2018-05-02/airscope_patched'
#CONFIG_PATH_X300 = '/home/hoops/airscope_2018-05-02/conf/airscope_pcell.conf'

AIRSCOPE_PATH_X300 = '/home/hoops/airscope_2019-09-02/airscope_patched'
CONFIG_PATH_X300 = '/home/hoops/airscope_2019-09-02/conf/airscope_pcell.conf'

USERNAME_X300 = 'hoops'
with open("passwd_hoops.txt", "r") as f2:
    PASSWD_X300 = f2.read()
LOGPATH_X300 = '/home/hoops/logs/%s' % (LOGDIR)

JHPC_HOST = '143.248.230.31'
JHPC_PORT = 22
AIRSCOPE_PATH_JHPC = '/home/hoops/airscope_2018-05-02/airscope_patched'
CONFIG_PATH_JHPC = '/home/hoops/airscope_2018-05-02/conf/airscope_scell.conf'
CONFIG_PATH_JHPC_PCELL = '/home/hoops/airscope_2018-05-02/conf/airscope_pcell.conf'
USERNAME_JHPC = 'hoops'
with open("passwd_dkay.txt", "r") as f3:
    PASSWD_JHPC = f3.read()
LOGPATH_JHPC = '/home/hoops/logs/%s' % (LOGDIR)


SUPC_HOST = '143.248.230.30'
SUPC_PORT = 22
AIRSCOPE_PATH_SUPC = '/home/hoops/airscope_2019-09-02/airscope_patched'
CONFIG_PATH_SUPC = '/home/hoops/airscope_2019-09-02/conf/airscope_pcell.conf'
CONFIG_PATH_SUPC_PCELL = '/home/hoops/airscope_2019-09-02/conf/airscope_pcell.conf'
USERNAME_SUPC = 'hoops'
with open("passwd_dkay.txt", "r") as f3:
    PASSWD_SUPC = f3.read()
LOGPATH_SUPC = '/home/hoops/logs/%s' % (LOGDIR)



# ==========================================================================
# PCELL, SCELL configuration
# ==========================================================================
#HOST_PCELL = X300_HOST
#PORT_PCELL = X300_PORT
#USERNAME_PCELL = USERNAME_X300
#PASSWD_PCELL = PASSWD_X300
#LOGPATH_PCELL = LOGPATH_X300

HOST_PCELL = SUPC_HOST
PORT_PCELL = SUPC_PORT
USERNAME_PCELL = USERNAME_SUPC
PASSWD_PCELL = PASSWD_SUPC
LOGPATH_PCELL = LOGPATH_SUPC


#HOST_PCELL = JHPC_HOST
#PORT_PCELL = JHPC_PORT
#USERNAME_PCELL = USERNAME_JHPC
#PASSWD_PCELL = PASSWD_JHPC
#LOGPATH_PCELL = LOGPATH_JHPC

HOST_SCELL = EKPC_HOST
PORT_SCELL = EKPC_PORT
USERNAME_SCELL = USERNAME_EKPC
PASSWD_SCELL = PASSWD_EKPC
LOGPATH_SCELL = LOGPATH_EKPC

#HOST_SCELL = JHPC_HOST
#PORT_SCELL = JHPC_PORT
#USERNAME_SCELL = USERNAME_JHPC
#PASSWD_SCELL = PASSWD_JHPC
#LOGPATH_SCELL = LOGPATH_JHPC

# ==========================================================================

# ========================= CONFIGURATION (Server) ==================================
# airscope binary and conf file location
#AIRSCOPE_PATH_PCELL = AIRSCOPE_PATH_X300
#CONFIG_PATH_PCELL = '/home/hoops/airscope_2018-05-02/conf/airscope_pcell.conf'

AIRSCOPE_PATH_PCELL = AIRSCOPE_PATH_SUPC
CONFIG_PATH_PCELL = CONFIG_PATH_SUPC

#AIRSCOPE_PATH_PCELL = AIRSCOPE_PATH_EKPC
#CONFIG_PATH_PCELL = CONFIG_PATH_EKPC

#AIRSCOPE_PATH_PCELL = AIRSCOPE_PATH_JHPC
#CONFIG_PATH_PCELL = CONFIG_PATH_JHPC_PCELL

# AIRSCOPE_PATH_SCELL = '/home/dkay/airscope_2018-05-02/airscope'
# CONFIG_PATH_SCELL = '/home/dkay/airscope_2018-05-02/conf/airscope_scell.conf'

AIRSCOPE_PATH_SCELL = AIRSCOPE_PATH_EKPC
CONFIG_PATH_SCELL = CONFIG_PATH_EKPC
# ==========================================================================

# ========================= Cell Phone dependent value =====================
# coordination
X_MAX=1072.5
Y_MAX=1912.5
# ==========================================================================

def cmd(cmd, interval=1):
    #print(cmd)
    os.system("adb shell " + cmd)
    time.sleep(interval)

def close_all_app():
    cmd("input keyevent KEYCODE_APP_SWITCH", interval=1)
    cmd("input tap 550 1870", interval=1)
    #cmd("input keyevent KEYCODE_DPAD_DOWN", interval=3)
    #cmd("input keyevent KEYCODE_ENTER", interval=3)
    #cmd("input keyevent KEYCODE_HOME")

def close_all_app_vert():
    cmd("input keyevent KEYCODE_APP_SWITCH", interval=1)
    cmd("input tap 970 990", interval=1)
    #cmd("input keyevent KEYCODE_DPAD_DOWN", interval=3)
    #cmd("input keyevent KEYCODE_ENTER", interval=3)
    #cmd("input keyevent KEYCODE_HOME")

def toggle_airplane(interval=10):
    cmd("am start -a android.settings.AIRPLANE_MODE_SETTINGS", 5)
    print('click to go airplane mode')
    cmd("input tap 960 280", interval) # Go to airplane
    print('click to come back from airplane mode')
    cmd("input tap 960 280", interval) # Re attach
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

def capture_screen(fname='screenshot.png'):
    # Capture the screenshot and pull it to PC
    cmd("/system/bin/screencap -p /sdcard/%s" % fname)
    os.system("adb pull /sdcard/%s" % (fname))
    return fname

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

def change_resolution(resolution=None):
    if resolution is None:
        print("No resolution is given")
        exit()
        return

    # Click setup!
    target_img_fname = "screenshot_setup_%d.png"
    match_img_fname = "resolution_imgbase/edge_setup_icon.png"
    setup_cord = None
    cnt = 0
    while setup_cord is None:
        if cnt > 10:
            print("[-] Error in change_resolution")
            return None

        cmd("input tap 400 152", interval=0.3) # click video first time
        cmd("input tap 1034 152", interval=0.3) # click video setting
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
    match_img_fname = "resolution_imgbase/edge_%dp.png" % (resolution)
    capture_screen(target_img_fname)
    cord = match_image(target_img_fname, match_img_fname)
    if cord is None:
       print("[-] There is no %dp resolution. Please check." % (resolution))
       return False
    else:
       cmd("input tap %d %d" % cord) # click target resolution
       return True

def open_youtube(name, link, resolution=None, interval=60):
    # Wait 5 second to load youtube app.
    cmd("am start -a android.intent.action.VIEW -d \"%s\"" % link, interval=5)
    sleep_time = 5

    # This takes about 5 seconds
    if resolution:
        status = change_resolution(resolution)
        if status is None:
            close_all_app()
            return False

        elif status == False:
            with open("no_resolution_list.txt", "a") as f:
                f.write(', '.join([name, link, str(resolution)]) + '\n')

        if name == "baby" and status == True:
            print(name, link, resolution)
            send_mail("check this!", ', '.join([name, link, str(resolution)]) + '\n')
        sleep_time += 5

    # Wait leftover interval before closing the video
    if interval > sleep_time:
        time.sleep(interval-sleep_time)

    close_all_app()
    return True

def open_dummy_youtube():
    # This link is only used for resolution change.
    dummy_link = "https://youtube.com/watch?v=o7mbYHwwcLE"
    open_youtube('dummy', dummy_link, RESOLUTION, interval=10)

def flush_youtube():
    cmd("pm clear com.google.android.youtube", interval=3)
    dummy_link = "https://youtube.com/watch?v=o7mbYHwwcLE"
    open_youtube('dummy', dummy_link, None, interval=10)
    open_dummy_youtube()

def login_netflix():
    dummy_link = "http://www.netflix.com/watch/70300666"
    # First start is used for logging in to Netflix after flushing the cache and user data.
    cmd("am start -a android.intent.action.VIEW \"%s\"" % dummy_link, interval=10)
    close_all_app()

def open_netflix(link, interval=60):
    link = link.replace('title', 'watch')
    # Actual loading the video.
    cmd("am start -a android.intent.action.VIEW -d \"%s\"" % link, interval)

    # move the playtime to the end of the movie.
    cmd("input tap 1600 850", interval=0.5)
    cmd("input swipe 1600 850 1900 850 500", interval=10)
    close_all_app_vert()

    return True

def flush_netflix():
    cmd("pm clear com.netflix.mediaclient", interval=3)
    login_netflix()

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

def start_log(name):
    btn_logging = (139, 107)
    click(btn_logging)
    write(name + '\n')
    write('y')

def end_log(name):
    btn_export= (480, 216)
    click(btn_export)
    btn_logname = (735, 748)
    click(btn_logname)
    write(name + '\n')
    btn_export = (742, 850)
    click(btn_export, interval=10)
    btn_ok = (944, 939)
    click(btn_ok)

    btn_logging = (139, 107)
    click(btn_logging)
    write('y')

    btn_clear = (780, 220)
    click(btn_clear)
    write('y')

def send_log(log_fname):
    transport = paramiko.Transport((HOST_PCELL, PORT_PCELL))
    transport.connect(username=USERNAME_PCELL, password=PASSWD_PCELL)
    sftp = paramiko.SFTPClient.from_transport(transport)
    logs = glob.glob('dmlogs/%s*' % (log_fname))
    for log in logs:
        #print(log)
        sftp.put(log, LOGPATH_PCELL + '/' + log.replace('dmlogs\\', ''))
        os.remove(log)
    sftp.close()
    transport.close()

def shell(cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST_PCELL, username=USERNAME_PCELL, password=PASSWD_PCELL)
    ssh.exec_command(cmd)
    time.sleep(3)
    ssh.close()

def send_mail(title, body):
    shell('python3 send_mail.py "%s" "%s"' % (title, body))

def kill_airscope_pcell():
    print('kill airscope @ PCELL')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST_PCELL, username=USERNAME_PCELL, password=PASSWD_PCELL)
    ssh.exec_command("ps aux | grep airscope | awk '{print $2}' | xargs sudo -S kill -9")
    #stdin.write(PASSWD_PCELL + "\n")
    #stdin.flush()
    time.sleep(3)
    #stdin, stdout, stderr = ssh.exec_command("ps aux | grep airscope | awk '{print $2}' | xargs sudo -S kill -9")
    #stdin, stdout, stderr = ssh.exec_command("ps aux | grep airscope | awk '{print $2}' | xargs sudo -S kill -9")
    #stdin.write(PASSWD_PCELL + "\n")
    #stdin.flush()
    #stdin, stdout, stderr = ssh.exec_command("ps aux | grep airscope | awk '{print $2}' | xargs sudo -S kill -9")
    #stdin.write(PASSWD_PCELL + "\n")
    #stdin.flush()
    ssh.close()

def kill_airscope_scell():
    if RF_NUM == 1:
        return

    print('kill airscope @ SCELL')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST_SCELL, username=USERNAME_SCELL, password=PASSWD_SCELL)
    ssh.exec_command("ps aux | grep airscope | awk '{print $2}' | xargs sudo -S kill -9")
    #stdin.write(PASSWD_SCELL + "\n")
    #stdin.flush()
    time.sleep(3)
    ssh.close()

def move_logfile_pcell(name):
    print('sudo cp "/tmp/airscope_pcell.csv" "%s/airscope_pcell_%s.csv"' % (LOGPATH_PCELL, name))
    print('sudo cp "/tmp/airscope_pcell.log" "%s/airscope_pcell_%s.log"' % (LOGPATH_PCELL, name))
    print('sudo cp "/tmp/airscope_pcell.pcap" "%s/airscope_pcell_%s.pcap"' % (LOGPATH_PCELL, name))
    
    #ssh = paramiko.SSHClient()
    #ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #ssh.connect(HOST_PCELL, username=USERNAME_PCELL, password=PASSWD_PCELL)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST_PCELL, username=USERNAME_PCELL, password=PASSWD_PCELL)

    ssh.exec_command('sudo cp "/tmp/airscope_pcell.csv" "%s/airscope_pcell_%s.csv"' % (LOGPATH_PCELL, name))
    ssh.exec_command('sudo cp "/tmp/airscope_pcell.log" "%s/airscope_pcell_%s.log"' % (LOGPATH_PCELL, name))
    ssh.exec_command('sudo cp "/tmp/airscope_pcell.pcap" "%s/airscope_pcell_%s.pcap"' % (LOGPATH_PCELL, name))
    time.sleep(5)
    ssh.close()

def move_logfile_scell(name):
    print('sudo cp "/tmp/airscope_scell.csv" "%s/airscope_%s.csv"' % (LOGPATH_SCELL, name))
    print('sudo cp "/tmp/airscope_scell.log" "%s/airscope_%s.log"' % (LOGPATH_SCELL, name))
    print('sudo cp "/tmp/airscope_scell.pcap" "%s/airscope_%s.pcap"' % (LOGPATH_SCELL, name))
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST_SCELL, username=USERNAME_SCELL, password=PASSWD_SCELL)
 
    ssh.exec_command('sudo cp "/tmp/airscope_scell.csv" "%s/airscope_scell_%s.csv"' % (LOGPATH_SCELL, name))
    ssh.exec_command('sudo cp "/tmp/airscope_scell.log" "%s/airscope_scell_%s.log"' % (LOGPATH_SCELL, name))
    ssh.exec_command('sudo cp "/tmp/airscope_scell.pcap" "%s/airscope_scell_%s.pcap"' % (LOGPATH_SCELL, name))
    time.sleep(5)
    ssh.close()

def stop_airscope_ca(name):
    print('kill airscope and move log file')
    kill_airscope_pcell()

    if RF_NUM == 2:
        kill_airscope_scell()

    print('Move logfiles')
    move_logfile_pcell(name)

    if RF_NUM == 2:
        move_logfile_scell(name)

    time.sleep(3)

def start_airscope_pcell(name):
    print ('start airscope pcell')

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST_PCELL, username=USERNAME_PCELL, password=PASSWD_PCELL)
    
   # sudo_cmd = 'sudo -S {0} "{1}" >/dev/null 2>&1'.format(AIRSCOPE_PATH_PCELL, CONFIG_PATH_PCELL)
    sudo_cmd = 'sudo -S {0} "{1}" > {3}/{2}_stdout.out  2> {3}/{2}_stderr.out'.format(AIRSCOPE_PATH_PCELL, CONFIG_PATH_PCELL, name, LOGPATH_PCELL)
    
    if HOST_PCELL == EKPC_HOST:
        print ("move to airscope dir")
        sudo_cmd = "cd /home/dkay/airscope_2019-09-02; " + sudo_cmd

    if HOST_PCELL == SUPC_HOST:
        print ("move to airscope dir")
        sudo_cmd = "cd /home/hoops/airscope_2019-09-02; " + sudo_cmd


    print (sudo_cmd)
    stdin, stdout, stderr = ssh.exec_command(sudo_cmd)
    stdin.write(PASSWD_PCELL + "\n")
    stdin.flush()
    ssh.close()

def start_airscope_scell():
    if RF_NUM == 1:
        return

    print ('start airscope scell')

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST_SCELL, username=USERNAME_SCELL, password=PASSWD_SCELL)
    
    sudo_cmd = 'sudo -S {0} "{1}" >/dev/null 2>&1'.format(AIRSCOPE_PATH_SCELL, CONFIG_PATH_SCELL)
    cmd = '{0} "{1}" >/dev/null 2>&1'.format(AIRSCOPE_PATH_SCELL, CONFIG_PATH_SCELL)
    stdin, stdout, stderr = ssh.exec_command(sudo_cmd)
    stdin.write(PASSWD_SCELL + "\n")
    stdin.flush()
    ssh.close()

def main():
    # get links
    links = get_links(VIDEO_LINK)
    pp.pprint (links)
#
    # move to optis screen
    print ("It starts after 5 seconds. Please move to optis-s screen")
    time.sleep(5)
#
    btn_clear = (780, 220)
    click(btn_clear)
    write('y')

    for idx in range(START_IDX, START_IDX + TRIAL):
        # first clear youtube cache and setup default resolution
        #flush_youtube()
        # first clear netflix cache and login with default account
        flush_netflix()

        print ("Open Netflix link")

        def start_test(idx, name, link):
            kill_airscope_pcell()
#            kill_airscope_scell()
#            
            print ('===============================')
            print (idx, name, link)
            print ('===============================')
            name = name.strip()
            link = link.strip()
            log_fname ='%s_%03d' % (name, idx)
#
            print ("\n\nToggle airplane")
            toggle_airplane()
#
            print ("\n\nStart airscope")
            start_airscope_pcell(log_fname)
            #time.sleep(2)
#            start_airscope_scell()
            print ("\n\nWait for booting airscope (15sec)")            
            time.sleep(10)
#
            print ("Start dm log")
            start_log(log_fname)
#            time.sleep(3)

#            print ("Open Youtube link")
#            status = open_youtube(name, link, RESOLUTION, interval=LOGGING_TIME)
            print ("Open Netflix link")
            status = open_netflix(link, interval=LOGGING_TIME)

            print ("Stop airscope")
            stop_airscope_ca(log_fname)
            
            print ("End log")
            end_log(log_fname)

            print ("Send log")
            send_log(log_fname)

            return status

        for name, link in links:
            name = name.strip()
            link = link.strip()
            cnt = 1

            print(idx, name, link)
            status = start_test(idx, name, link)
            while status != True:
                cnt += 1
                send_mail("Error in testing ()", "%d-th trial, %d, %s, %s, %d" % (cnt, idx, name, link, RESOLUTION))
                print("[+] Restarting ... %d th trial" % cnt)
                #flush_youtube()
                flush_netflix()
                status = start_test(idx, name, link)


main()

