"""
set the logging file
"""
import logging

"""
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This will get logged to a file')


logging.basicConfig(filename='app.log', filemode='w',format='%(process)d-%(levelname)s-%(message)s')
logging.warning('This is a Warning')
"""

logging.basicConfig(filename='app.log', filemode='w',format='%(asctime)s - %(message)s')
logging.warning('Admin logged in')
