import copy
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

import tqdm
from tqdm.auto import tqdm as tqdm_
import numpy as np
from sklearn.base import BaseEstimator
from mne.decoding.receptive_field import _delay_time_series
from sklearn.linear_model import RidgeCV

from ..data import Data
from ..utils import _parse_outstruct_args


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull.
    This is used to suppress tqdm outputs from fitting, which produce
    a progress bar to sys.stderr."""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


class TRF(BaseEstimator):
    '''
    Class for fitting temporal receptive field (TRF) models
    to one or more targets at a time. These can be
    encoding models (stimulus-to-brain) or decoding (brain-to-stimulus) models.
    This estimator used internally is sklearn.linear_model.RidgeCV() but
    can be specified as any estimator which implements the basic sklearn estimator
    API, including all estimators in sklearn.linear_model.

    Please see the :ref:`TRF example notebooks <STRF examples>` for more detailed
    tutorials which show how to train, test, and inspect TRF and STRF models.
    
    Parameters
    ----------
    tmin : float
        The starting lag (inclusive), in seconds (or samples if ``sfreq`` == 1).
    tmax : float
        The ending lag (inclusive), in seconds (or samples if ``sfreq`` == 1).
        Must be > tmin.
    sfreq : int
        The sampling frequency used to convert times into samples.
    estimator : sklearn.linear_model instance, default=RidgeCV()
        Estimator to use for each target. Must be compatible with sklearn API
        for Regressors (i.e. have fit(), predict(), score() methods, and a coef_ attribute).
        The default is sklearn.linear_model.RidgeCV() with alphas=np.logspace(-2, 5, 6),
        scoring='r2', cv=5. This trains ridge regularized regressors with built-in cross-validation
        over the regularization parameter using 5-fold cross validation.
    verbose : int, default=1
        Level of printing output desired.
        0 prints nothing, 1 prints only a single tqdm progress bar for
        the fitting over the n-outputs in .fit(), and 2 prints information about
        cross-validation, such as the alpha value chosen and the corresponding scores,
        during the fitting procedure for each output.

    Attributes
    ----------
    coef_ : np.ndarray, shape (n_targets[, n_features_y], n_features_X, n_lags)

    Notes
    -----
    For a causal system, the encoding model would have significant
    non-zero values only at positive lags. In other words, lags point
    backwards in time relative to the input, so positive lags correspond
    to previous time samples, while negative lags correspond to future
    time samples. In most cases, an encoding model should use tmin=0
    and tmax>0, while a decoding model should use tmin<0 and tmax=0.
    
    '''
    def __init__(self,
                 tmin,
                 tmax,
                 sfreq,
                 estimator=None,
                 verbose=1):
        
        if tmin >= tmax:
            raise ValueError(f'tmin must be less than tmax, but got {tmin} and {tmax}')
        
        self.sfreq = float(sfreq)
        self.tmin = tmin
        self.tmax = tmax
        if estimator is None:
            self.estimator = RidgeCV(alphas=np.logspace(-2, 5, 6), cv=5)
        else:
            self.estimator = estimator
        self.verbose = verbose
        
    
    @property
    def _smin(self):
        return int(round(self.tmin * self.sfreq))

    @property
    def _smax(self):
        return int(round(self.tmax * self.sfreq)) + 1
    
    @property
    def _ndelays(self):
        return self._smax - self._smin
        
    def _delay_and_reshape(self, X, y=None):
        """Delay and reshape the variables.
        X and y should be arrays.
        X has shape (time, features)
        """
        if not X.ndim == 2:
            raise ValueError(f'Each trial input must be 2 dimensional but got trial with shape {X.shape}')
        # X is now shape (n_times, n_feats, n_delays)
        X = _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                               fill_mean=False)
        x_shape = X.shape
        X = X.reshape(x_shape[0], -1)
            
        return X, y

    def fit(self, data=None, X='aud', y='resp'):
        '''
        Fit a multi-output model to the data in X and y, which contain multiple trials.
        
        Parameters
        ----------
        data : naplib.Data instance, optional
            Data object containing data to be normalized in one of the field.
            If not given, must give the X and y data directly as the ``X``
            and ``y`` arguments. 
        X : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as predictor in the regression. Should
            be of shape (time, num_features).
            If a string, it must specify one of the fields of the Data
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances which will be concatenated over to compute
            normalization statistics.
        y : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as target(s) in the regression. Once arranged,
            should be of shape (time, num_targets[, num_features_y]).
            If a string, it must specify one of the fields of the Data
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances.
        
        Returns
        -------
        self : returns an instance of self
        
        '''
        
        X_, y_ = _parse_outstruct_args(data, copy.deepcopy(X), copy.deepcopy(y))
        
        if y_[0].ndim == 1:
            y_ = [yy[:,np.newaxis] for yy in y_]
        
        self.ndim_y_ = y_[0].ndim
        
        X_delayed, y_delayed = [], []
        
        self.X_feats_ = X_[0].shape[-1]

        for xx, yy in zip(X_, y_):
            X_tmp, y_tmp = self._delay_and_reshape(xx, yy)
            X_delayed.append(X_tmp)
            y_delayed.append(y_tmp)
        
        X_delayed = np.concatenate(X_delayed, axis=0)
        y_delayed = np.concatenate(y_delayed, axis=0)
        
        self.n_targets_ = y_[0].shape[1]
        self.y_feats_ = y_[0].shape[-1] if self.ndim_y_==3 else 1
        
        # for each target variable, fit a TRF model
        self.models_ = [copy.deepcopy(self.estimator) for _ in range(y_delayed.shape[1])]
        
        if self.verbose >= 1:
            disable_tqdm = False
        else:
            disable_tqdm = True

        for target_idx in tqdm_(range(y_delayed.shape[1]), disable=disable_tqdm):
            y_thistarget = y_delayed[:,target_idx]
            if self.verbose >= 2:
                print(f'Fitting model for output variable {target_idx}...')
            # if self.ndim_y_ == 2:
            #     y_thistarget = y_thistarget[:,np.newaxis]
            self.models_[target_idx].fit(X_delayed, y_thistarget)
             
        return self
    
    @property
    def coef_(self):
        if not hasattr(self, 'ndim_y_'):
            raise ValueError(f'Must call fit() first before accessing coef_ attribute.')
        coefs_ = []
        for mdl in self.models_:
            if self.ndim_y_ == 3:
                coefs_.append(mdl.coef_.reshape(1, self.y_feats_, self.X_feats_, self._ndelays))
            else:
                coefs_.append(mdl.coef_.reshape(1, self.X_feats_, self._ndelays))
                
        return np.concatenate(coefs_, axis=0) # shape (num_outputs[, target_features], X_features, lags)
            
    def predict(self, data=None, X='aud'):
        '''
        Parameters
        ----------
        data : naplib.Data object, optional
            Data object containing data to be normalized in one of the field.
            If not given, must give the X and y data directly as the ``X``
            and ``y`` arguments. 
        X : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as predictor in the regression. Once arranged,
            should be of shape (time, num_features).
            If a string, it must specify one of the fields of the Data
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances which will be concatenated over to compute
            normalization statistics.
        
        Returns
        -------
        y_pred : np.ndarray or list of np.ndarrays, each with shape (time, num_targets[, num_features_y])
            Predicted target for each trial in X.
        '''
        
        if not hasattr(self, 'models_'):
            raise ValueError(f'Must call .fit() before can call .predict()')
        
        X = _parse_outstruct_args(data, X)
        
        X_delayed = []

        for xx in X:
            X_tmp, _ = self._delay_and_reshape(xx)
            X_delayed.append(X_tmp)
            
        y_pred = [[] for _ in range(len(X))]
        for ii, x_trial in enumerate(X_delayed):
            for mdl in self.models_:
                tmp_pred = mdl.predict(x_trial)
                if self.ndim_y_ == 2:
                    tmp_pred = tmp_pred.reshape(-1,1)
                else:
                    tmp_pred = tmp_pred.reshape(tmp_pred.shape[0],1,-1)
                y_pred[ii].append(tmp_pred)
            y_pred[ii] = np.concatenate(y_pred[ii], axis=1)
        
        return y_pred
    
    def score(self, data=None, X='aud', y='resp'):
        '''
        Get scores from predictions of the model. This uses the defualt
        score() method from the estimator used.
        
        Parameters
        ----------
        data : naplib.Data object, optional
            Data object containing data to be normalized in one of the field.
            If not given, must give the X and y data directly as the ``X``
            and ``y`` arguments. 
        X : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as predictor in the regression. Once arranged,
            should be of shape (time, num_features).
            If a string, it must specify one of the fields of the Data
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances which will be concatenated over to compute
            normalization statistics.
        y : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as target(s) in the regression. Once arranged,
            should be of shape (time, num_targets[, num_features_y]).
            If a string, it must specify one of the fields of the Data
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances which will be concatenated over to compute
            normalization statistics.

        Returns
        -------
        scores : np.array of float, shape (n_targets,)
            The scores estimated by the model for each output.
        '''
        
        if not hasattr(self, 'models_'):
            raise ValueError(f'Must call .fit() before can call .score()')
        
        X_, y_ = _parse_outstruct_args(data, copy.deepcopy(X), copy.deepcopy(y))    

        X_delayed, y_delayed = [], []
        
        if y_[0].ndim == 1:
            y_ = [yy[:,np.newaxis] for yy in y_]
        

        for xx, yy in zip(X_, y_):
            X_tmp, y_tmp = self._delay_and_reshape(xx, yy)
            X_delayed.append(X_tmp)
            y_delayed.append(y_tmp)
        
        X_delayed = np.concatenate(X_delayed, axis=0)
        y_delayed = np.concatenate(y_delayed, axis=0)
        
        scores = []
        for target_idx in range(y_delayed.shape[1]):
            y_thistarget = y_delayed[:,target_idx]
            scores.append(self.models_[target_idx].score(X_delayed, y_thistarget))
        
        return np.array(scores)

    def corr(self, data=None, X='aud', y='resp'):
        '''
        Get correlation coefficient from predictions of the model.
        
        Parameters
        ----------
        data : naplib.Data object, optional
            Data object containing data to be normalized in one of the field.
            If not given, must give the X and y data directly as the ``X``
            and ``y`` arguments. 
        X : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as predictor in the regression. Once arranged,
            should be of shape (time, num_features).
            If a string, it must specify one of the fields of the Data
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances which will be concatenated over to compute
            normalization statistics.
        y : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as target(s) in the regression. Once arranged,
            should be of shape (time, num_targets[, num_features_y]).
            If a string, it must specify one of the fields of the Data
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances which will be concatenated over to compute
            normalization statistics.

        Returns
        -------
        corrs : np.array of float, shape (n_targets,)
            The correlations of the predictions by the model for each output.
        '''
        
        if not hasattr(self, 'models_'):
            raise ValueError(f'Must call .fit() before can call .score()')
        
        X_, y_ = _parse_outstruct_args(data, copy.deepcopy(X), copy.deepcopy(y))    

        X_delayed, y_delayed = [], []
        
        if y_[0].ndim == 1:
            y_ = [yy[:,np.newaxis] for yy in y_]
            
        for xx, yy in zip(X_, y_):
            X_tmp, y_tmp = self._delay_and_reshape(xx, yy)
            X_delayed.append(X_tmp)
            y_delayed.append(y_tmp)
        
        X_delayed = np.concatenate(X_delayed, axis=0)
        y_delayed = np.concatenate(y_delayed, axis=0)
                
        scores = []
        for target_idx in range(y_delayed.shape[1]):
            y_thistarget = y_delayed[:,target_idx].reshape(-1,)
            pred = self.models_[target_idx].predict(X_delayed).reshape(-1,)
            scores.append(np.corrcoef(y_thistarget, pred)[0,1])
        
        return np.array(scores)
        