import datetime
import threading
import os


class Logger:
    def __init__(self, name, logdir='stdout'):
        self.name = name
        self.logdir = logdir
        self.logfile = os.path.join(logdir, '{}.log'.format(name))

    def info(self, str):
        if self.logdir != 'stdout':
            with open(self.logfile, 'a') as fw:
                fw.write(str + '\n')
        else:
            print(str)

    def log(self, tags, message, *args):
        tag = ":".join(tags)
        header = "{}[{}:{}]:".format(datetime.datetime.now(), self.name, tag)
        message = str(message)
        message = message.format(*args)
        if self.logdir != 'stdout':
            with open(self.logfile, 'a') as fw:
                fw.write(header + message + '\n')
        else:
            print(header + message)

    def error(self, message="", *args):
        self.log(["ERROR"], message, *args)

    def info(self, message="", *args):
        self.log(["INFO"], message, *args)

    def warn(self, message="", *args):
        self.log(["WARNING"], message, *args)

    def debug(self, message, *args):
        self.log(["DEBUG"], message, *args)


logger = Logger(threading.current_thread().name)
