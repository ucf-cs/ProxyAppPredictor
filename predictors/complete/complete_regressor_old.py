'''

Created by Kenneth Lamar at 9/1/2022
Based on work by by Alexander Goponenko

'''
import numpy as np
from __future__ import division
from sklearn.base import BaseEstimator, RegressorMixin
from recorder_mem import Recorder

PARAM_NAME = 'duration'

class CompleteRegressor(BaseEstimator, RegressorMixin):
  """
  This predictor has 16 kind of tags, with pre-set priorities (see get_tags).
  The prediction is calculated using highest priority record that is found.
  During update of the prediction, all 16 records corresponding to the record are updated.

  NOTE: modified to exclude the base predictor
      also it can return None instead of prediction
  """

  def get_tags(self, job):
    return (
      '{}|{}|{}|{}'.format(job["job_name"], job["user"], job["timelimit"], job["nodes"]),
      '{}|{}|{}|{}'.format(job["job_name"], job["user"], job["timelimit"], ''),
      '{}|{}|{}|{}'.format(job["job_name"], job["user"], '', job["nodes"]),
      '{}|{}|{}|{}'.format(job["job_name"], job["user"], '', ''),
      '{}|{}|{}|{}'.format(job["job_name"], '', job["timelimit"], job["nodes"]),
      '{}|{}|{}|{}'.format(job["job_name"], '', job["timelimit"], ''),
      '{}|{}|{}|{}'.format(job["job_name"], '', '', job["nodes"]),
      '{}|{}|{}|{}'.format(job["job_name"], '', '', ''),

      '{}|{}|{}|{}'.format('', job["user"], job["timelimit"], job["nodes"]),
      '{}|{}|{}|{}'.format('', job["user"], job["timelimit"], ''),
      '{}|{}|{}|{}'.format('', job["user"], '', job["nodes"]),
      '{}|{}|{}|{}'.format('', job["user"], '', ''),
      '{}|{}|{}|{}'.format('', '', job["timelimit"], job["nodes"]),
      '{}|{}|{}|{}'.format('', '', job["timelimit"], ''),
      '{}|{}|{}|{}'.format('', '', '', job["nodes"]),
      # '{}|{}|{}|{}'.format('', '', '', ''),
    )

  def get_tags_names(self):
    """ Helper function useful to generate data related to the performance of the predictor

    :return: list of names that help identify the tag "king"
    """
    return (
      '{}|{}|{}|{}'.format("job", "user", "time", "node"),
      '{}|{}|{}|{}'.format("job", "user", "time", ''),
      '{}|{}|{}|{}'.format("job", "user", '', "node"),
      '{}|{}|{}|{}'.format("job", "user", '', ''),
      '{}|{}|{}|{}'.format("job", '', "time", "node"),
      '{}|{}|{}|{}'.format("job", '', "time", ''),
      '{}|{}|{}|{}'.format("job", '', '', "node"),
      '{}|{}|{}|{}'.format("job", '', '', ''),

      '{}|{}|{}|{}'.format('', "user", "time", "node"),
      '{}|{}|{}|{}'.format('', "user", "time", ''),
      '{}|{}|{}|{}'.format('', "user", '', "node"),
      '{}|{}|{}|{}'.format('', "user", '', ''),
      '{}|{}|{}|{}'.format('', '', "time", "node"),
      '{}|{}|{}|{}'.format('', '', "time", ''),
      '{}|{}|{}|{}'.format('', '', '', "node"),
      # '{}|{}|{}|{}'.format('', '', '', ''),
    )

  def normalizeRecord(record):
    return record if record else (0.0, 0.0, 0.0, 0.0)

  def update_param(self, job, param_name, value, var=None):
    """updates prediction  for 'param_name' of the 'job'

    NOTE: new version to calculate variance

    Updates are made for all tags for the job returned by self.get_tags.

    :param job:  the job record (usually a dataframe row or dictionary)
    :param param_name: the name of the parameter to update
    :param value: the average resource utilization for the job
    :param var: the variance of the resource utilization for the job
    :return: tuple (new average, new StDev)
             which is ignored by simulator but may be useful for analysis
    """
    if var is not None:
      raise NotImplementedError()

    alpha = self.decays[param_name]
    res = None
    point_weight = value if self.use_weights else 1
    for tag in self.get_tags(job):
      pAvg, pSquare, pCount, pSum = normalizeRecord(self.db.getRecord(tag, param_name))
      nCount = point_weight + (1 - alpha) * pCount
      nSum = value * point_weight + (1 - alpha) * pSum
      nAvg = nSum / nCount
      nSquare = value*value * point_weight + (1 - alpha) * pSquare
      if nAvg * nSum / nSquare > 1.0000000001:
        print("Problem")
        print(alpha)
        print(pAvg, pSquare, pCount, pSum)
        print(nAvg, nSquare, nCount, nSum)
        print(nAvg * nSum)
        print(nSquare - nAvg * nSum)
        assert False
      self.db.saveRecord(tag, param_name, nAvg, nSquare, nCount, nSum)
      if res is None:
        if nCount > 1:
          res = (nAvg, self._var(nAvg, nCount, nSquare, nSum)**0.5)
        else:
          res = (None, None) # FIXME: what is the best base case?
    return res or (None, None)

  def predict_requirements(self, job, param, real=None):
    """makes prediction for 'param' for the 'job'

    :param job: the job record (usually a dataframe row or dictionary)
    :param param: the name of the parameter
    :param real: real value for the job (not used here - for compatibility with 'ideal' predictor)
    :return: the predicted value
    """
    tags = self.get_tags(job)
    for tag in tags:
      res = self.db.getRecord(tag, param)
      if res:
        pAvg, pSquare, pCount, pSum = res
        if self.sigma_factor is None:
          return pAvg
        if pCount > 1:
          pVar = self._var(pAvg, pCount, pSquare, pSum)
          return pAvg + self.sigma_factor * pVar**0.5
    return None


  def _var(self, avg, count, square, pSum):
    assert count > 1, "need more than 1 measurement"
    pVar = (square - avg * pSum) / (count - 1)
    if pVar < 0:
      if pVar < -1:
        raise Exception()
      pVar = 0.0
    return pVar

  def __init__(self, options = None):
    alpha = options["scheduler"]["predictor"].get('decay', 0.2)
    sigma_factor = options["scheduler"]["predictor"].get('sigma_factor', None)
    use_weights = options["scheduler"]["predictor"].get('use_weights', False)
    decays = {PARAM_NAME : alpha}
    recorder = Recorder()

    self.db = Recorder()
    self.decays = {PARAM_NAME : alpha}
    self.sigma_factor = options["scheduler"]["predictor"].get('sigma_factor', None)
    self.use_weights = options["scheduler"]["predictor"].get('use_weights', False)

    jrp_predictor = JRPBase(recorder, decays, [PARAM_NAME], sigma_factor, use_weights)
    super(PredictorComplete, self).__init__(jrp_predictor, PARAM_NAME)

  def fit(self, X, y):
    """
    Add a job to the learning algorithm.
    Called when a job end.

    :return: (optionally; only needed for evaluation)
              newly calculated (prediction, error)
    """
    job = {
      "actual_run_time": y[-1],
      "user_estimated_run_time": X[-1],
      "predicted_run_time": None
    }
    jrp_job = self._make_jrp_job(job)
    value = job.actual_run_time
    res = self.predictor.update_param(jrp_job, self.param, value)
    if res is None:
      return job.user_estimated_run_time, job.user_estimated_run_time
    else:
      avg, var = res
      return (avg or job.user_estimated_run_time, var or job.user_estimated_run_time)

  def predict(self, X):
    """
    Modify the predicted_run_time of job.
    Called when a job is submitted to the system.
    the predicted runtime should be an int!
    """
    jrp_job = self._make_jrp_job(job)
    result = self.predictor.predict_requirements(jrp_job, self.param, job.actual_run_time)
    # print("Got prediction: {}".format(result))
    if result is None:
      job.predicted_run_time = job.user_estimated_run_time
    else:
      result = int(round(result))
      job.predicted_run_time = min(result, job.user_estimated_run_time)
    return