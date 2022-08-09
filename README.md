# Description
This repository contains the tools used in our paper,
Watching the Watchers: Practical Video Identification Attack in LTE Networks,
accepted at USENIX Security 2022.

For more details, please check [our
paper](https://www.usenix.org/conference/usenixsecurity22/presentation/bae)

# Dataset

**Table 1: Dataset summary**
|Dataset|# of Videos|# of Traces|Description|
|:---:|:---:|---:|---|
|YouTube100|100|29,715|YouTube Top 100 in [1]|
|Netflix|22|1,001|Netflix Top 50|
|Amazon|32|1,210|Prime Video 32|
|YouTubeCA|100|7,383|YouTube Top 100 in [1] (w/ CA)|
|YouTube200|200|6,424|YouTube Top 101-300 in [1]|
|Web|-|268|Visiting randomly selected 45 websites from Alexa Top 50 Website [2]|
|Teleconf|-|201|Google Meet|

[1]: Most viewed music videos of all time. https://kworb.net/youtube/topvideos.html.

[2]: Katharina Kohls, David Rupprecht, Thorsten Holz, and Christina P̈opper. Lost Traffic Encryption: Fingerprinting LTE/4G Traffic on Layer Two. ACM WiSec19

**Table 2: # of Traces used in the evaluation (YouTube100 videos)**
|MNO|Video Quality|# of Traces||MNO|Video Quality|# of Traces||MNO|Video Quality|# of Traces|
|:---:|:---:|---:|-|:---:|:---:|---:|-|:---:|:---:|---:|
|A|480p|3,184||B|480p|3,376||C|480p|3,262|
|A|720p|3,196||B|720p|3,318||C|720p|3,218|
|A|1080p|3,645||B|1080p|3,223||C|1080p|3,293|



# How to use

## Environment Setup

You need to prepare at least one labtop, one cellphone, and one Linux server.


### Labtop Setup

Set up Innowireless DM tool on a labtop
and connect a cellphone used to collect data.
Most of the Python scripts will be run on this labtop.


### Linux Server Setup

Set up airscope on a Linux server.
We use airscope to capture radio signals in the same way as an adversary.

Additionally, please copy `send_mail.py` to the Linux Server and install `smtpd`
on the server. This script will be run to notify you when experiments are
completed via email.

TODO: Add description sudo setup for `mv` and `kill`


### Configuration
Many configuration variables depend on your experimental environment.
Please see [config.py](config.py) and modify the variables appropriately.



<!--- 
## Data Collection

To collect dataset, we XXX

## Classification?

adsfasdfasdf


# Issues

### Tested environment

For data collection, we ran the Python scripts on a labtop running Winodws 10
and airscope on a server running Linux XXXXX.
For running DM, XXX
-->


# Authors
This project has been conducted by the below authors at KAIST.
* [Sangwook Bae](https://sites.google.com/site/sangwookbae89)
* Mincheol Son
* [Dongkwan Kim](https://0xdkay.me/)
* CheolJun Park
* Jiho Lee
* [Sooel Son](https://sites.google.com/site/ssonkaist/home)
* [Yongdae Kim](https://syssec.kaist.ac.kr/~yongdaek/)

# Citation
We would appreciate if you consider citing [our
paper](https://www.usenix.org/conference/usenixsecurity22/presentation/bae) when
using the tools.
```bibtex
@inproceedings{bae:2022:watching,
  author = {Sangwook Bae, Mincheol Son, Dongkwan Kim, CheolJun Park, Jiho Lee, Sooel Son, and Yongdae Kim},
  title = {Watching the Watchers: Practical Video Identification Attack in LTE Networks},
  booktitle = {Proceedings of the 31st USENIX Security Symposium (Security)},
  mon = aug,
  year = 2022,
  address = {Boston, MA}
}
```
