#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
from logging import INFO, Formatter, StreamHandler, getLogger


class Log():
    logger = {}
    
    @classmethod
    def getLogger(cls, name="root", no_handler=False):
        if name in cls.logger:
            return cls.logger[name]

        # logger
        logger = getLogger(name)
        logger.setLevel(INFO)
        
        if not no_handler:
            sh = StreamHandler()
            sh.setLevel(INFO)
            hf = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            sh.setFormatter(hf)
            logger.addHandler(sh)
        
        cls.logger[name] = logger
        return logger
