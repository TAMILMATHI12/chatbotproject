# Python code to illustrate Sending mail from
# your Gmail account
import smtplib
def process(remail,msg):
    
    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login("gowsalya12345321@gmail.com", "gowsalya54321")

    # message to be sent
    message = msg

    # sending the mail
    s.sendmail("sender_email_id", remail, message)

    # terminating the session
    s.quit()
