import sys
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

def send_mail(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = '[Video] ' + subject
    msg['From'] = SERVER_ADDRESS
    msg['To'] = TO_ADDRESS
    msg['Cc'] = CC_ADDRESS
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()

if __name__ == '__main__':
    # argv[1]: subject
    # argv[2]: body
    subject = sys.argv[1]
    body = sys.argv[2]
    body += '\n' + str(datetime.now()) + '\n'
    send_mail(subject, body)
