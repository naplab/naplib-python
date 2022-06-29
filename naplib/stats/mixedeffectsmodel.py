import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearMixedEffectsModel():
    '''
    A linear mixed effects model which can be used for main effect plots.
    A mixed effects model is useful when the data has some non-independence
    whose effect should be accounted for separately from the effects of
    the independent variables. For example, if half of the data comes
    from subject A and the other half from subject B, and you want to
    know the main effects of other predictor features.
    
    Parameters
    ----------
    alpha : float, default=0.05
        Alpha level for confidence interval.
    zscore_x : bool, default False
        Whether or not to zscore the columns of X when calling .fit().
        
    Attributes
    ----------
    varnames : list
        List of variable names
    mixedlm : statsmodels.regression.mixed_linear_model.MixedLM object
        Object which performs mixed linear model fitting
    mixedlm_results : statsmodels.regression.mixed_linear_model.MixedLMResults object
        Object which contains outputs from fitting.
    params : pd.DataFrame
        Coefficients for each variable.
    pvalues : pd.DataFrame
        Pvalues for each param.
    conf_int : pd.DataFrame
        Confidence interval around each param.
    rsquared : np.float
        R-squared value.
    
    '''
    def __init__(self, alpha=0.05, zscore_x=False):
        self._alpha = alpha
        self._zscore_x = zscore_x
        self.varnames = []
        self._isfit = False
        
    def fit(self, X, y, random_effect=None, varnames=None):
        '''
        Parameters
        ----------
        X : np.array, shape (num_samples, num_features)
        y : np.array, shape (num_samples,)
        random_effect : np.array, shape (num_samples,), optional
            If given, used as a random effect in the model. For example,
            to use group identity as a random effect, this should be an array
            of integers specifying what group each sample belongs to. The
            values do not matter, only the categorical groups they form.
        varnames : list, optional, default None
            List of variable names, must be length (num_features+1) or
            (num_features + num_features_r + 1), giving the name
            for each feature in X, each feature in random_effect (if given)
            as well as a name for the predicted output in y.
            
        Returns
        -------
            model : returns an instance of self
        '''
        if random_effect is not None:
            if random_effect.ndim == 1:
                random_effect = random_effect[:, np.newaxis]
            elif random_effect.shape[1] > 1:
                raise ValueError(f'Only 1 random effect variable is currently supported, but got {random_effect.shape[1]}')
        
        
        if varnames is None:
            varnames = [f'feature-{i}' for i in range(X.shape[1])]
            varnames.append('y')
        if random_effect is None and len(varnames) != X.shape[1] + 1:
            raise Exception(f'Error: incorrect length of input "varnames". Must include name for each feature in X as well as a name for predicted y.')
        if random_effect is not None and len(varnames) != X.shape[1] + random_effect.shape[1] + 1:
            raise Exception(f'Error: incorrect length of input "varnames". Must include name for each feature in X, each feature in random_effect,'
                            f' as well as a name for predicted y.')
        self.varnames = varnames[:X.shape[1]] + varnames[-1:]
        
        # remove spaces and dashes because they break the smf.mixedlm formula
        varnames_formula = varnames.copy()
        for bad_char in [' ', '-']:
            varnames_formula = [v.replace(bad_char, '') for v in varnames_formula]
        
        warnings.simplefilter('ignore', ConvergenceWarning)
        
        if self._zscore_x:
            X = (X - X.mean(0)) / X.std(0)
            
        self._y = y
        
        if random_effect is None:
            df_tmp = pd.DataFrame(np.concatenate((X, y.reshape(-1,1)), axis=1), columns=varnames_formula)
            formula = f'{varnames_formula[-1]} ~ {varnames_formula[0]}'
            for varname in varnames_formula[1:-1]:
                formula += f' + {varname}'
            mixedlm = smf.mixedlm(formula=formula, data=df_tmp, groups=[1 for _ in range(df_tmp.shape[0])])

        else:
            df_tmp = pd.DataFrame(np.concatenate((X, random_effect, y.reshape(-1,1)), axis=1), columns=varnames_formula)
            formula = f'{varnames_formula[-1]} ~ {varnames_formula[0]}'
            for varname in varnames_formula[1:X.shape[1]]:
                formula += f' + {varname}'
            mixedlm = smf.mixedlm(formula=formula, data=df_tmp, groups=varnames[-2], re_formula='1')

            
        mixedlm_results = mixedlm.fit()
        
        self.mixedlm = mixedlm
        self.mixedlm_results = mixedlm_results
        
        self.params = mixedlm_results.params
        self.pvalues = mixedlm_results.pvalues
        self.conf_int = mixedlm_results.conf_int(alpha=self._alpha)
        
        self._isfit = True
        
        return self
        
    def get_model_params(self):
        '''
        Get model params after fitting.
        
        Returns
        -------
        param_dict : dict   
            dict of model params, pvalues, and confidence intervals for each variable.
        '''
        if not self._isfit:
            raise Exception('Must call .fit() before trying to acces model params.')
        return {'params': np.array(self.params)[1:-1], 'pvalues': np.array(self.pvalues)[1:-1],
                'conf_int': np.array(self.conf_int)[1:-1,:]}
    
    def plot_effects(self, ax=None, plus_minus_colors=None, line_alpha=0.6, markersize=8, center_zero=True, print_ylabels=True):
        '''
        Create main effects plot.
        
        Parameters
        ----------
        ax : pyplot axes, optional, default=gca()
        plus_minus_colors : list or np.array, length=2, default red, blue
            Colors to use for the effect lines when they are significantly positive or negative, respectively.
        line_alpha : float, default=0.06
            alpha value for lines
        markersize : float, default=8
            Size of marker at each effect
        center_zero : bool, default=True
            Whether to put 0 in the center of x-axis.
        print_ylabels : bool, default=True
            Whether to print y labels
        '''
        if ax is None:
            ax = plt.gca()
        
        if plus_minus_colors is None:
            plus_minus_colors = np.array([[255., 44, 0], [97., 164, 205]]) / 255 # blue and red
        
        model_params = self.get_model_params()
        
        
        for effect_idx in range(len(model_params['params'])):
            # check if this effect is significant
            if model_params['params'][effect_idx] > 0 and model_params['conf_int'][effect_idx].min() > 0:
                color_to_use = plus_minus_colors[0]
            elif model_params['params'][effect_idx] < 0 and model_params['conf_int'][effect_idx].max() < 0:
                color_to_use = plus_minus_colors[1]
            else:
                color_to_use = [0.5, 0.5, 0.5]
                
            ax.plot(model_params['conf_int'][effect_idx,:], effect_idx * np.ones_like(model_params['conf_int'][effect_idx-1,:]), color=color_to_use, alpha=line_alpha)
            ax.plot(model_params['params'][effect_idx], np.array([effect_idx]), marker="o", markersize=markersize, markeredgecolor=color_to_use, markerfacecolor=color_to_use)
        
        ax.set_ylim([-0.5, 1.5])
        if center_zero:
            xabs_max = abs(max(ax.get_xlim(), key=abs)) * 1.1
            ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
        
        ylims = ax.get_ylim()
        ax.set_ylim([-0.5, len(model_params['params'])-0.5])
        ylims = ax.get_ylim()
        ax.vlines(0, ylims[0], ylims[1],'k',linestyles='dashed', linewidth=1, alpha=0.5)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        if print_ylabels:
            ax.tick_params(top=False, bottom=True, left=False, right=False, labelleft=True, labelbottom=True)
        else:
            ax.tick_params(top=False, bottom=True, left=False, right=False, labelleft=False, labelbottom=True)
        
        ax.set_yticks([nn for nn in range(len(model_params['params']))])
        ax.set_yticklabels(self.varnames[:-1])
        
        return ax
    
    def summary(self):
        '''
        Print summary of model.
        '''
        print(self.mixedlm_results.summary())
        
    @property
    def rsquared(self):
        '''r-squared : float'''
        rss_tot = np.sum(np.square(self._y-self._y.mean()))
        rss_res = np.sum(np.square(self.mixedlm_results.resid))
        return 1.0 - rss_res/rss_tot
