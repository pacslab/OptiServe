import logging


class Logger:
    def __init__(self, 
                 directory='./', 
                 filename='log.log'):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=f'{directory}/{filename}', level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        
    def info(self, message):
        self.logger.info(message)
        
    
    def debug(self, message):
        self.logger.debug(message)
        
    
    def warning(self, message):
        self.logger.warning(message)
        
    
    def error(self, message):
        self.logger.error(message)