import logging


class Logger:
    def __init__(self, 
                 directory='./', 
                 filename='log.log'):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=f'{directory}/{filename}')
        
    def info(self, message):
        self.logger.info(message)
        
    
    def debug(self, message):
        self.logger.debug(message)
        
    
    def warning(self, message):
        self.logger.warning(message)
        
    
    def error(self, message):
        self.logger.error(message)