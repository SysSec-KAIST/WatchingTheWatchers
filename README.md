# Description
This repository contains the tools used in our paper,
Watching the Watchers: Practical Video Identification Attack in LTE Networks,
accepted at USENIX Security 2022.

For more details, please check [our
paper](https://www.usenix.org/conference/usenixsecurity22/presentation/bae)

# Dataset

Bring the table here.


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
