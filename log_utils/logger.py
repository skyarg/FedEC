#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import logging.handlers
import logging.config
import os
from utils.options import args_parser
import time

class InfoFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.INFO:
            return super().filter(record)
        else:
            return 0

args = args_parser()

localtime = time.localtime(time.time())
year = localtime[0]
month = localtime[1]
day = localtime[2]
hour = localtime[3]
logname = args.function+"_"+str(month)+"-"+str(day)+"_"+str(args.dataset)+"_"+str(args.num_users)+"_"+str(args.epochs)+"_"+str(args.frac)+"_"+str(args.seed)
log_dir = "./logging/"+logname+"/"

log_dict = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            'format': '%(asctime)s | %(process)d | %(levelname)s | %(filename)s | %(funcName)s | %(lineno)d | %(message)s'
        }
    },

    'filters': {
        'info_filter': {
            '()': InfoFilter,
        },
    },

    'handlers': {
        'info': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': os.path.join(log_dir, 'loss.log'),
            # 'maxBytes': 1024 * 1024 * 100,
            'formatter': 'standard',
            "encoding": "utf-8",
            # 'filters': ['info_filter'],
            # 'backupCount': 50
        }
    },

    'loggers': {
        'info': {
            'handlers': ['loss'],
            'level': 'INFO'
        }
    },
}

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

open(log_dict['handlers']['info']['filename'], 'a').close()

logging.config.dictConfig(log_dict)

info_logger = logging.getLogger('info')