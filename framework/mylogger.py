'''
Created on 2020/07/09

@author: ukai
'''
from datetime import datetime


class MyLogger(object):
    '''
    classdocs
    '''
    
    def __init__(self, console_print = True):
        self.console_print = console_print

    def info(self, agent, buildParameter, environment, epoch, trainer):
                 
        txt = "{0}, epoch: {1:5d}, build label: {2}".format(
            datetime.now()
            , epoch
            , buildParameter.label)
        
        if self.console_print:
            print(txt)