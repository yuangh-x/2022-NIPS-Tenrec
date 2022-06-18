import os
import torch
# from abc import

class LoggerService(object):
    def __init__(self, train_loggers=None, val_loggers=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)