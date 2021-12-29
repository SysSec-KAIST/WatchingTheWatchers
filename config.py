# ==========================================================================
# ======================= CONFIGURATION (Video Logging) ====================
# ==========================================================================
START_IDX = 1  # Set up this index to continue the sample collection
TRIAL = 40  # Number of samples to collect

# Select right resolution. this can be 480, 720, 1080
RESOLUTION = 1080
# RESOLUTION = 720
# RESOLUTION = 480

LOGGING_TIME = 280
# LOGGING_TIME = 1000

VIDEO_LINK = "youtube_links_u.txt" # deprecated. Give links by parameter

# ==========================================================================
# ========================= Email Setup in (Server) ========================
# ==========================================================================
TO_ADDRESS = '0xdkay@gmail.com'
CC_ADDRESS = 'baesangwook89@gmail.com'
SERVER_ADDRESS = 'video@injector.syssec.kr'

# ==========================================================================
# ========================= CONFIGURATION (Server) =========================
# ==========================================================================
# PC IP addr, port number
# airscope binary and conf file location
# ==========================================================================

# ==========================================================================
# ========================= PCELL, SCELL configuration =====================
# ==========================================================================
# number of cells (1: PCELL, 2: SCELL, 3: SCELL2)
RF_NUM = 1

HOST_IP = "143.248.230.30"
PORT_ADDR = 22
USERNAME_= "asdf"
PASSWD = "asdf"

# airscope binary and conf file location
AIRSCOPE_BIN_PATH = "/home/asdf/airscope/airscope"
AIRSCOPE_CONFIG_PATH = "/home/asdf/airscope/conf/airscope_pcell.conf"
AIRSCOPE_LOG_DIR = "/home/{}/logs/".format(USERNAME)


# ==========================================================================
# ========================= Cell Phone dependent value =====================
# ==========================================================================
# coordination
#X_MAX = 1072.5
#Y_MAX = 1912.5
CORD_CLOSE_APP = (550, 1870)
CORD_CLOSE_APP_VERT = (970, 990)
# CORD_ENTER_AIRPLANE = (850, 2500)
CORD_TOGGLE_AIRPLANE = (960, 280)
CORD_YOUTUBE_CENTER = (400, 152)
CORD_YOUTUBE_SETTING = (1034, 152)
CORD_YOUTUBE_SECRET = (550 1374)
CORD_NETFLIX_LOGIN = (380, 950)
CORD_NETFLIX_PLAYSTART = (1600, 850)
CORD_NETFLIX_PLAYEND = (1900, 850)
CORD_WEB_1 = (97, 1275)
CORD_WEB_2 = (535, 1820)
CORD_WEB_3 = (150, 1820)
CORD_MEET_1 = (150, 1820)
CORD_MEET_2 = (350, 1685)
NETFLIX_SWIPE_DURATION = 500

# Select resolution prefix. if you use different cellphone, you need to capture
# the image on the phone first.
RESOLUTION_PREFIX = "edge"
#RESOLUTION_PREFIX = "galaxy_note8"


# ==========================================================================
# ========================= Laptop dependent value =========================
# ==========================================================================
BTN_LOGGING = (130, 100)
BTN_EXPORT = (500, 220)
BTN_EXPORT_LOGNAME = (950, 750)
BTN_EXPORT_EXPORT = (950, 850)
BTN_EXPORT_OK = (950, 940)
BTN_CLEAR = (780, 210)


# ==========================================================================
# ========================= dummy links ====================================
# ==========================================================================
# these are random links not included in the dataset
YOUTUBE_DUMMY = "https://youtube.com/watch?v=o7mbYHwwcLE"
NETFLIX_DUMMY = "http://www.netflix.com/watch/70300666"

