import time
from datetime import datetime

def StatusPrint(*messages,isTemp=False,printTime=True):
    if printTime:
        messages = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),) + messages

    print('\033[2K',end='')
    print(*messages,end='\r' if isTemp else '\n')
