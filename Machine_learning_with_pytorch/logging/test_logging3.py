"""
The test logging for exceptions 
"""



import logging

a = 5
b = 0

try:
  c = a / b
except Exception as e:

    # the true and False will set to log only the line 
  logging.error("Exception occurred", exc_info=False)