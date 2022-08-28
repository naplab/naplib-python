import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mpltext

from naplib.stats import LinearMixedEffectsModel


@pytest.fixture(scope='module')
def data():
    '''
    Generates independent variables and dependent variables where there is
    a random effect of subject identity.
    There are n_subjects, and each has N_per_subject data points.
    '''
    # The true effects of each feature will be 1 for feature 1 and -2 for feature 2
    N_per_subject=500
    n_subjects=5
    betas = np.array([1, -2])
    X = [] # independent variables
    Y = [] # dependent variable
    subject_ID = [] # subject ID random effect variable
    data_mean = np.array([0., 0.]).reshape(1,-1)
    rng = np.random.default_rng(1)
    subj_betas = np.array([1, 0])
    for i in range(n_subjects):
        X_thissubject = rng.uniform(size=(N_per_subject,2)) + data_mean
        Y_thissubject = X_thissubject @ betas + data_mean @ subj_betas - 3*data_mean[0,0] + rng.normal(scale=0.15, size=(N_per_subject,))
        X.append(X_thissubject)
        Y.append(Y_thissubject)
        subject_ID.append(i*np.ones_like(Y_thissubject))
        data_mean += np.array([1, 0]).reshape(1,-1)

    return {'x': np.concatenate(X), 'y': np.concatenate(Y), 'ID': np.concatenate(subject_ID)}

def test_with_random_effect(data):
    model = LinearMixedEffectsModel()
    varnames = ['Feature 1', 'Feature 2', 'Subject', 'Output Variable']
    model.fit(data['x'], data['y'], random_effect=data['ID'], varnames=varnames)
    params = model.get_model_params()
    expected_weights = np.array([ 0.99110757, -2.00410041])
    expected_confint = np.array([[ 0.97061759,  1.01159756],
                                 [-2.02424875, -1.98395207]])
    assert np.allclose(params['params'], expected_weights, atol=1e-7)
    assert np.allclose(params['pvalues'], np.array([0,0]), atol=1e-7)
    assert np.allclose(params['conf_int'], expected_confint)

    assert np.allclose(model.rsquared, 0.9910296832649543, atol=1e-9)

def test_without_random_effect(data):
    model = LinearMixedEffectsModel()
    varnames = ['Feature 1', 'Feature 2', 'Output Variable']
    model.fit(data['x'], data['y'], varnames=varnames)
    params = model.get_model_params()
    expected_weights = np.array([-0.92052704, -2.02810964])
    expected_confint = np.array([[-0.93620308, -0.90485101],
                                 [-2.10595005, -1.95026923]])
    assert np.allclose(params['params'], expected_weights, atol=1e-7)
    assert np.allclose(params['pvalues'], np.array([0,0]), atol=1e-7)
    assert np.allclose(params['conf_int'], expected_confint)

    assert np.allclose(model.rsquared, 0.8657349455141483, atol=1e-9)

def test_plot_of_main_effects(data):
    model = LinearMixedEffectsModel()
    varnames = ['Feature 1', 'Feature 2', 'Subject', 'Output Variable']
    model.fit(data['x'], data['y'], random_effect=data['ID'], varnames=varnames)
    params = model.get_model_params()
    expected_weights = np.array([ 0.99110757, -2.00410041])
    expected_confint = np.array([[ 0.97061759,  1.01159756],
                                 [-2.02424875, -1.98395207]])
    
    fig, ax = plt.subplots(1,1)
    ax = model.plot_effects(ax=ax)

    # check plotted main effects 
    assert np.allclose(ax.lines[0].get_data()[0], np.array([0.97061759, 1.01159756]), atol=1e-7) # first conf int
    assert np.allclose(ax.lines[1].get_data()[0], np.array([0.99110757]), atol=1e-7) # first main effect
    assert np.allclose(ax.lines[2].get_data()[0], np.array([-2.02424875, -1.98395207]), atol=1e-7) # second conf int
    assert np.allclose(ax.lines[3].get_data()[0], np.array([-2.00410041]), atol=1e-7) # second main effect

    ylbl = ax.get_yticklabels()

    assert ylbl[0].get_position()==(0,0)
    assert ylbl[0].get_text()=='Feature 1'
    assert ylbl[1].get_position()==(0,1)
    assert ylbl[1].get_text()=='Feature 2'

def test_get_model_params_before_fit():
    model = LinearMixedEffectsModel()
    with pytest.raises(Exception):
        model.get_model_params()

