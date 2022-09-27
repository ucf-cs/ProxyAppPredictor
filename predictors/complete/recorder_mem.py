'''
In-memory recorder.
  Stores the records in RAM instead of a database.

Created on Apr 21, 2020

@author: alex
'''

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class Recorder(object):

  def __init__(self):
    """ This recorder keeps all the records in a dict in RAM

    :param path: not used for this recorder
    """

    self.db = {}


  def getRecord(self, variety_id, param):
    """ Retrieves record for given tag (variety_id) and parameter name

    :param variety_id: tag of the job group
    :param param: parameter (i.e. the name of the predicted value)
    :return: the record as a tuple (avg, var, w_count, w_sum)
             or None if such record not found
    :rtype: tuple or None
    """
    key = (variety_id, param)
    record = self.db.get(key, None)
    if not record:
      return None
    else:
     return record

  def saveRecord(self, variety_id, param, avg, var, w_count, w_sum):
    """Stores the record

    :param variety_id: tag of the job group
    :param param: parameter (i.e. the name of the predicted value)
    :param avg: average value of the parameter
    :param var: variance for the parameter
    :param w_count: "count of weights" for calculating the average value
    :param w_sum:  "sum of weighted values" for calculating the average value
    :return: nothing
    """
    logger.debug("saving variety_id: \"%s\", parameter: \"%s\", avg: %f, var: %f, count: %f, sum: %f",
                 variety_id, param, avg, var, w_count, w_sum)
    key = (variety_id, param)
    self.db[key] = (avg, var, w_count, w_sum)
