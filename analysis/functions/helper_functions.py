import os
import errno
from glob import glob
from os.path import split, exists
import numpy as np
from functools import lru_cache

import pymc3 as pm
import theano as T
import theano.tensor as tt
import arviz as az
import pandas as pd
from scipy import stats


def save_trace(name, trace, path_to_folder):
    try:
        os.makedirs(f'{path_to_folder}/{name}')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    trace.to_netcdf(
        f'{path_to_folder}/{name}/trace.nc'
    )
                    

def load_trace(name, path_to_folder):
    az.rcParams["data.load"] = 'eager'
    returndict = dict()
    returndict['trace'] = az.from_netcdf(
        f'{path_to_folder}/{name}/trace.nc'
    )
    az.rcParams["data.load"] = 'lazy'
    
    # TODO: I am ignoring models for now because
    # loading them with pickle causes an error.
    # They can be redefined quite simply directly in the code as needed.
    # But leave it open for loading model in the future
    
    return returndict


def get_traces(path_to_folder):
    
    traces = dict()
    for path in glob(path_to_folder+'/*'):
        name = split(path)[-1]
        print('Getting ', name)
        try:
            # this line is rather dreadful but it'll have to do
            traces[name] = eval('trace_'+name)
            print(f'Objects for {name} already defined, using \n')
        except NameError:
            completepath = path+'/trace.nc'
            print(f'Objects for {name} not yet defined, attempting from file')
            try:
                simulation_data = load_trace(
                    name,
                    path_to_folder
                )
                traces[name] = simulation_data['trace']
                # models[name] = simulation_data['model']
                print(f'Objects for {name} loaded from file\n')
            except FileNotFoundError:
                print(f'File for {name} not found, giving up.\n')
    return traces


def load_argstrengths(savename):
    if savename is not None:
        # try to get argstrengths from file
        if exists(savename):
            with open(savename, 'rb') as openf:
                argstrengths = np.load(openf)
            # return the previously stored argstrengths
            return argstrengths
    else:
        return None


def array_to_hashable(arr):
    return arr.tobytes()


@lru_cache(maxsize=None)
def internal_verify(q2, arr_hashable, n_answers, n_students):
    arr = np.frombuffer(arr_hashable, dtype=int).reshape(-1,n_students)
    # '...Q of the questions ADJ'
    if q2 == 'none':
        value = (arr == 0)
    elif q2 == 'some':
        value = (arr > 0)
    elif q2 == 'most':
        value = (arr > n_answers/2)
    elif q2 == 'all':
        value = (arr == n_answers)
    else:
        raise ValueError('q2 not recognized!')
    return value.sum(axis=-1)


def verify(q1, q2, adj, arr, n_answers=None, n_students=None):
    """
    Check literal truth of utterance for an arr.
    wrt utterance "q1 of the students got q1 of the answers adj"
    NOTE: arr can be 2d
    
    Parameters
    ----------
    q1, q2, adj: string
        The bits needed to construct the sentence
    arr: 1-d or 2-d array of ints
        An array containing the number of questions
        that the students got right.
        Dimensions: (# answers, # students) | (# answers)
    Returns
    -------
    1-d array
        Array containing for each set of answers
        whether it verifies the utterance.
    """
    
    if n_answers is None:
        n_answers = arr.max()
    if n_students is None:
        n_students = arr.shape[1]
    
    arr = np.array(arr)
    
    # '...of the questions AJD'
    if adj == 'wrong':
        arr = n_answers - arr
    
    # Convert arr to hashable
    arr_hashable = array_to_hashable(arr)
    
    ntrue = internal_verify(q2, arr_hashable, n_answers, n_students)
    
    # 'Q of the students got Q of the questions ADJ'
    if q1 == 'none':
        result = ntrue == 0
    elif q1 == 'some':
        result = ntrue > 0
    elif q1 == 'most':
        result = ntrue > n_students/2
    elif q1 == 'all':
        result = ntrue == n_students
    else:
        raise ValueError('q1 not recognized!')
    
    return result.astype(int)


def get_costs(utterances, costtype='positive'):
    """
    Used for the calculate_logp_data function
    Reproduces the utterance costs of the cogsci paper if
    costtype is 'positive'.
    Otherwise just returns uniform costs.
    """
    if costtype == 'positive':
        # utterances with a negative quantifier have cost 3, otherwise cost 0
        return np.any(utterances == 'none', axis=1).astype(int)*3
    elif costtype == 'uniform':
        # assume for now that cost is uniform
        return np.ones((1, len(utterances)))
    

def normalize(arr, axis=1):
    """
    Normalize arr along axis
    """
    return arr / arr.sum(axis, keepdims=True)


def softmax(x, axis=1):
    """
    Softmax function in numpy

    Parameters
    ----------
    x: array
        An array with any dimensionality
    axis: int
        The axis along which to apply the softmax
    Returns
    -------
    array
        Same shape as x
    """
    e_x = np.exp(x - np.max(x, axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def calculate_pragmatic_speaker(argumentative_strengths,
                                p_observation_given_utterance, costs,
                                alpha, beta, truth_matrix=0):
    """
    Calculate the probability of the pragmatic speaker producing each utterance
    given each observation.

    Parameters
    ----------
    argumentative_strengths: array
        The argumentative strength of each utterance
        for whatever aim the speaker has
        (NOTE: this can be positive or negative argstrengths,
        different speakers are calculated for the two cases)
    p_observation_given_utterance: array
        The probability of each observation given an utterance.
        Basically the literal listener.
        Dimensions: (possible utterances, possible observations)
        In Michael's implementation, this array 
        has all identical rows and the selection of 
        the incompatible combinations of utt and obs
        is done by the truth_matrix, which has a very small
        value (in log space) in order to not break the gradient
        when the logging of 0 happens in informativity.
    costs: array
        The cost of each utterance
    alpha, beta: float
        The parameters of the model.
        See explanation of the model for detail.
    truth_matrix: array
        See explanation of p_observation_given_utterance input.
        Keep it 0 if not using Michael's method.
    Returns
    -------
    array
        Dimensions (possible utterances, possible observations)
        The pragmatic speaker.
    """
    
    informativity = np.log(p_observation_given_utterance)
    utils = (
        truth_matrix +
             beta  * informativity
        + (1-beta) * argumentative_strengths[:,None]
        - costs.reshape(-1,1)
    )
    p_utterance_given_observation = softmax(alpha * utils, axis=0)
    return p_utterance_given_observation


def theano_normalize(tensor, axis):
    """
    For explanations of input/outputs,
    see documentation of normalize in functions
    """
    return tensor / tt.sum(tensor, axis=axis, keepdims=True)

def theano_softmax(tensor, axis):
    """
    For explanations of input/outputs,
    see documentation of softmax in functions
    """
    # NOTE: axis=0 only works for 2d arrays
    if axis==1:
        return tt.nnet.softmax(tensor)
    else:
        return tt.nnet.softmax(tensor.T).T


def u_o_array_to_df(array, possible_observations, possible_utterances):
    """
    This is a function to help visualize arrays in terms
    of actual possible_observations and possible_utterances
    rather than just indices.

    Parameters
    ----------
    array: 2d array
        Dimensions (possible utterances, possible observations)
    Returns
    -------
    df
        A df with the possible utterances as rows
        and the possible observations as columns.
    """
    return pd.DataFrame(
        array,
        index=['|'.join(u) for u in possible_utterances],
        columns=[str(o) for o in possible_observations]
    )


def theano_calculate_pragmatic_speaker(p_observation_given_utterance, 
                                       costs, alpha, 
                                       beta=None, argumentative_strengths=None):
    """
    This function is written in such a way that it accepts either:
        - alpha as a single value (completely pooled without argumentation)
        - alpha as 1-d tensor (hierarchical without argumentation)
        - alpha and beta as single values (completely pooled with argumentation)
        - both alpha and beta as 1d tensors (by-participant pooling with argumentation)
    If argumentation is modelled, you must specify BOTH beta and argumentative_strengths,
    otherwise neither.
    
    Parameters
    ----------
    argumentative_strengths: None or theano tensor
        Dims: (utterance)
    p_observation_given_utterance: tensor
        Dims: (utterance, observation)
    costs: tensor
        Dims: (utterance)
    alpha: theano float or tensor
        Dims: float or (participant)
    beta: None or theano float or tensor
        Dims: float or (participant)
    """
    
    assert not (
        (beta is None) ^ 
        (argumentative_strengths is None)
    ), "Specify both beta and argstrength or neither!"
    
    hierarchical = alpha.type.ndim > 0
    
    # if alpha is not just an int
    # we do both alpha and beta hierarchically
    if hierarchical:
        
        print('Defining hierarchical model')
        # reshape into (participant, utterance, observation)
        
        # if beta wasn't specified, it's just set to one
        # (this happens when there's no argumentation)
        # but still with shape (1,1,1) because it's hierarchical
        if beta is None:
            beta = np.array([[[1]]])
            argumentative_strengths = np.array([[[1]]])
        else: 
            beta = beta[:,None,None]
            try:
                # if it is a theano tensor
                argstrength_dim = argumentative_strengths.type.ndim
            except AttributeError:
                # if it a numpy array
                argstrength_dim = argumentative_strengths.ndim
                
            if argstrength_dim == 1:
                # argstrength had dimension (utterance)
                # this is the usual case
                argumentative_strengths = argumentative_strengths[None,:,None]
            elif argstrength_dim == 2:
                # argstrength had dimension (participant, utterance)
                # this is the case where each participant has a different
                # argstrength, e.g. the pragmatic argstrength case
                argumentative_strengths = argumentative_strengths[:,:,None]
            else:
                raise InputError('argstrength has strange shape!')
        
        alpha = alpha[:,None,None]
        costs = costs[None,:,None]
        p_observation_given_utterance = p_observation_given_utterance[None]
    
    else:
        print('Defining non-hierarchical model')
        # if beta wasn't specified, it means we don't have 
        # argumentation, so it's equivalent to just setting to 1
        if beta is None:
            beta = 1
            argumentative_strengths = np.array([[1]])
        else:
            argumentative_strengths = argumentative_strengths[:,None]
        
        # reshape into (utterance, observation)
        costs = costs[:,None]
    
    p_part = p_observation_given_utterance**(alpha*beta)
    weighted_argstrength = (1-beta) * argumentative_strengths
    unnorm_softmax = (
        p_part * tt.exp(alpha * ( weighted_argstrength - costs ))
    )
    p_utterance_given_observation = theano_normalize(
        unnorm_softmax, 
        axis= 1 if hierarchical else 0
    )
    return p_utterance_given_observation


def calculate_bayesian_posterior_pvalue(trace, pps, model, return_full=False, pointwise=False):
    """
    Calculates the Bayesian p-value with the loglikelihood as the 
    statistic of interest. 
    
    NOTE: In order to use this function, the data in the pymc3 model has to be 
    recorded in a pm.Data sharedvariable called 'observed'
    
    Suppose we already have a trace for the observed data. 
    We need to do two things to calculate the bayesian p-value:
    - Take sample of simulated datasets from the posterior
        - This can be done directly with the pm.sample_posterior_predictive function
    - For each posterior sample in the trace:
        - Calculate the probability of the statistic of choice 
            - for the simulated data from that sample
            - for the actual data
        - Record whether the statistic of the simulated data was more extreme 
          than the one of the actual data for that posterior sample.
    - Take the proportion out of all posterior samples.
    
    Here I am exploiting some less-known pymc3 things:
        - model.names_vars: dictionary with {variable name: variable}
        - pm.util.dataset_to_point_dict: function that turns an az trace into
          a list of dictionaries.
        - pm.datalogpt: theano tensor that contains the likelihood
        - pm.set_data: function that allows to change the theano shared tensor
          containing the data. 
    Technically, 
        - I loop through the posterior samples in the trace
        - I change the dataset to the simulated dataset for that posterior sample
        - I get the likelihood of that dataset for that posterior sample
        - I accumulate it in a list
    At the end, I compare the accumulated list with the likelihoods of the
    actual data stored in the trace (I double checked that it corresponds
    to the loglikelihood obtained with model.datalogpt)
    """
    # start by calculating the appropriate loglikelihood
    # for the actual data
    data_eval_loglik = trace.log_likelihood.to_array().values
        
    # list of dict, each dict is a posterior sample
    trace_ = pm.util.dataset_to_point_dict(trace.posterior)
    
    # to calculate this the graph needs to be built,
    # since this takes time I am doing it outside of the loop
    if pointwise:
        # NOTE: assumption that there's exactly one observed RV
        assert len(model.observed_RVs) == 1, 'Check observed'
        
        # model_logp_f returns elementwise logp
        model_logp_f = model.observed_RVs[0].logp_elemwiset.flatten().eval
        
        # get loglikelihood of actual data for each posterior sample 
        # directly from trace.
        # Dimensions of trace.log_likelihood are
        # (1, chain, sample, datapoint)
        # and remaining dimensions depend on the shape of the data.
        # data_evaluated_loglik get compared with sim_evaluated_loglik. 
        # sim_evaluated_loglik below has dimensions 
        # (posterior samples, datapoint).
        # I want to flatten data_eval_loglik to 
        # (posterior samples, datapoint) so it can be 
        # compared with sim_evaluated_loglik
        data_evaluated_loglik = data_eval_loglik.reshape(
            (-1, data_eval_loglik.shape[-1]) 
        )
        
    else:
        # model_logp_f returns total logp
        model_logp_f = model.datalogpt.eval
        # get total loglikelihood of actual data 
        # for each posterior sample directly from trace.
        # If there is only one observed, 
        # dimensions of trace.log_likelihood are
        # (1, chain, sample, datapoint)
        # so I need to sum over all but the first three dimensions
        axes_for_sum = np.arange(len(data_eval_loglik.shape))[3:]
        data_evaluated_loglik = data_eval_loglik.sum(tuple(axes_for_sum)).flatten()
        
    # deterministic vars cause
    # unused input error because they
    # make their parents unused.
    # To prevent this error, temporarily ignore
    # unused input
    T.config.on_unused_input = 'ignore'
    
    sim_evaluated_loglik = []
    # iterate over the posterior samples (for all free parameters)
    for i, vartrace in enumerate(trace_):
        # evaluate loglikelihood of posterior sampled utterances
        # change value of utterances SharedVariable (pymc3 Data object)
        # into the utterances vector
        
        # replace variable name with actual variable in dict keys
        vartrace = {
            model.named_vars[key]: value 
            for key, value 
            in vartrace.items()
        }
        # get posterior predictive sample of utterances
        # for that trace sample
        pm.set_data(
            {'observed': pps[i]}, 
            model
        )
        # get probability of observed (likelihood)
        # for that posterior sample
        sim_evaluated_loglik.append(model_logp_f(
            vartrace,
        ))        
        
        if i%100==0:
            print(i, end=', ')
    
    T.config.on_unused_input = 'raise'
    
    # actual observations
    real_data = trace.observed_data.to_dataframe().values.flatten()

    # give data_utterances its original value
    pm.set_data(
        {'observed': real_data}, 
        model
    )
    
    if return_full:
        return (np.array(sim_evaluated_loglik), data_evaluated_loglik)
    else:
        # check if the posterior predictive sample has a loglikelihood
        # as low or lower than the really observed data,
        # for each trace sample
        at_least_as_extreme = np.array(sim_evaluated_loglik) <= data_evaluated_loglik
        
        if pointwise:
            # dims before summing (trace sample, datapoint)
            # dims after summing (datapoint)
            return at_least_as_extreme.sum(0)/len(at_least_as_extreme)
        else:
            # dims before summing (trace sample)
            return at_least_as_extreme.sum()/len(at_least_as_extreme)
        

def parameter_recovery_MAP(model, nsamples, data):
    # Simulated datasets from the prior
    # Assuming that the prior is somewhat vague
    # this should give a good coverage of the parameter space
    with model:
        prior_pd = pm.sample_prior_predictive(nsamples)

    # Find the MAP of all parameters for each simulated dataset
    MAP_values = []
    for i, row in enumerate(prior_pd['utterances']):
        with model:
            pm.set_data({'observed': row})
            MAP_values.append(pm.find_MAP(progressbar=False))
        print(i, end=', ')
    pm.set_data({'observed': data.index_utterance}, model=model)
    return prior_pd, MAP_values


def from_posterior(param, samples, lb=False, ub=False):
    """
    Function to go from trace from previous fit
    to distribution to use as prior.
    Use gaussian kde for approximation.
    
    Parameters
    ----------
    param: string
        Name of parameter
    samples: array
        Array with the posterior values
    lb, ub: Bool
        Whether the parameter has an upper/lower bound.
        If true, lower bound is set to 0 and upper bound to 1.
    """
    
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(
        smin, 
        smax, 
        100
    )
    y = stats.gaussian_kde(samples)(x)

    min_x = x[0] - 3 * width
    if lb:
        # since the param is lower-bounded at 0, add this
        min_x = max(0.00001, min_x)
        
    max_x = x[-1] + 3 * width
    if ub:
        max_x = min(max_x, 0.99999)
        
    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([
        [min_x], 
        x,
        [max_x]
    ])
    
    y = np.concatenate([
        [0], 
        y, 
        [0]
    ])
    
    return pm.Interpolated(param, x, y)