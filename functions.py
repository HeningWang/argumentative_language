import pandas as pd
from scipy import stats
import numpy as np
from itertools import product
import os
import errno
import pickle
import xarray as xr


def save_model_and_trace(name, model, trace):
    
    try:
        os.makedirs(f'models_traces/{name}')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    with open(f'models_traces/{name}/model.pkl', 'wb') as openfile:
        pickle.dump(model, openfile)
    trace.to_netcdf(f'models_traces/{name}/trace.nc')
                    

def load_model_and_trace(name):
    
    returndict = dict()
    
    with xr.open_dataset(f'models_traces/{name}/trace.nc') as ds:
        returndict['trace'] = ds
        
    with open(f'models_traces/{name}/model.pkl', 'rb') as openfile:
        returndict['model'] = pickle.load(openfile)
    
    return returndict


def calculate_alpha(mu, rho):
    """
    calculate alpha parameter of beta
    distribution in terms of mu and rho
    """
    return (1-rho) * mu / rho


def calculate_beta(mu, rho):
    """
    calculate beta parameter of beta
    distribution in terms of mu and rho
    """
    return (1-rho) * (1-mu) / rho


def beta_binomial_pmf(mu, rho, n):
    """
    return pmf of beta-binomial in 
    with mu and rho parameterization
    """
    alpha = calculate_alpha(mu, rho)
    beta = calculate_beta(mu, rho)
    return stats.betabinom.pmf(
        k=np.arange(n), 
        n=n, 
        a=alpha, 
        b=beta
    )


def verify(q1, q2, adj, arr):
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
    
    arr = np.array(arr)
    
    if adj == 'wrong':
        arr = 12 - arr
    
    if q2 == 'none':
        value = (arr == 0)
    elif q2 == 'some':
        value = (arr > 0)
    elif q2 == 'most':
        value = (arr >= 6)
    elif q2 == 'all':
        value = (arr == 12)
    else:
        raise ValueError('q2 not recognized!')

    ntrue = value.sum(axis=-1)
    
    if q1 == 'none':
        result = ntrue == 0
    elif q1 == 'some':
        result = ntrue > 0
    elif q1 == 'most':
        result = ntrue > 2
    elif q1 == 'all':
        result = ntrue == 5
    else:
        raise ValueError('q1 not recognized!')
    
    return result.astype(int)


def get_and_clean_data():
    """
    Get the raw data and cleans it. 
    Returns various useful objects (see below).
    
    Returns
    -------
    tuple
        raw_data: pd df 
            Dataframe containing the full data 
        data: pd df 
            Dataframe containing the data without 
            incomplete and literally false responses
        possible_observations: 2-d array
            Array with the possible unique observations
            that participants actually saw in the experiment.
            Contains the number of correct answers
            of each student in each observation.
            Dimensions: (observations, students)
        possible_utterances: 2-d array
            Array with the possible unique utterances
            that the participants could produce in the experiment.
            Each utterance has three components: 
            (outer quant, inner quant, adjective)
            Dimensions: (utterances, utterance components)
    """
    raw_data = pd.read_csv('data_raw.csv')
    
    data = raw_data[[
        'condition', 'response', 'row_number', 'trial_name', 'prolific_id'
    ]]
    
    data = (
        data[raw_data['trial_name']=='main_trials']
        .reset_index(drop=True)
    )
    
    data['row_number'] = (
        data
        .row_number
        .str
        .split('|')
        .apply((
            lambda x: [int(a) for a in x]
        ))
    )

    data['response'] = (
        data
        .response
        .str
        .split('|')
    )
    
    # Some responses are not recorded completely
    # So I need to exclude them
    print(
        (1-data.response.apply(lambda x: '' not in x)).sum(), 
        ' were excluded because incompletely recorded'
    )
    
    data = data[data.response.apply(lambda x: '' not in x)]
    
    qs = ['none', 'some', 'most', 'all']
    adjs = ['right', 'wrong']

    possible_observations = np.array([
            a.split('|')
            for a in raw_data['row_number'].unique()
        ], 
        dtype=int
    )

    possible_utterances = np.array(list(product(qs, qs, adjs)))

    index_observations_data = data.row_number.apply(
        lambda observation: np.argwhere((possible_observations==observation).all(1)).flatten()[0]
    )
    data['index_observation'] = index_observations_data

    index_utterance_data = data.response.apply(
        lambda utterance: np.argwhere((possible_utterances==utterance).all(1)).flatten()[0]
    )
    data['index_utterance'] = index_utterance_data
    data['condition'] = (data.condition=='high').astype(int)
    
    # exclude from the data literally wrong responses
    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])
    index_literally_true = utterance_observation_compatibility[
        data.index_utterance, 
        data.index_observation
    ].astype(bool)
    print(
        (1-index_literally_true).sum(),
        ' were excluded because literally false'
    )
    data = data.iloc[index_literally_true]
    
    # record the index of the participant
    _, participant_id = np.unique(data['prolific_id'], return_inverse=True)
    data['id'] = participant_id
    data = data.drop('prolific_id', axis=1)
    
    return raw_data, data, possible_observations, possible_utterances


def calculate_arg_deltas(argumentative_strengths, possible_observations, possible_utterances):
    """
    Parameters
    ----------
    argumentative_strengths: array 
        see return value of calculate_argumentative_strengths
    possible_observations, possible_utterances: 2-d array
        See return values of get_and_clean_data function
    Returns
    -------
    array
        Difference in argument strength between 
        the chosen utterance and the most argstrong one
        possible given the observation
    """
    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])
    argstrengths_masked = np.where(
        utterance_observation_compatibility,
        argumentative_strengths[:,None],
        -np.inf
    )
    argdeltas = argstrengths_masked.max(0, keepdims=True) - argstrengths_masked
    return argdeltas
    
    
def calculate_info_deltas(possible_utterances, possible_observations):
    """
    Parameters
    ----------
    possible_utterances, possible_observations: arrays
        See return values of get_and_clean_data function
    Returns
    -------
    array
        Difference in informativeness between 
        the chosen utterance and the most informative one
        possible given the observation
    """
    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])
    p_observation_given_utterance = normalize(utterance_observation_compatibility,1)
    informativity = np.log(p_observation_given_utterance)
    infodeltas = informativity.max(0, keepdims=True) - informativity
    # infodeltas has shape (# utterances, # observations)
    return infodeltas


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


def normalize(arr, axis=1):
    """
    Normalize arr along axis
    """
    return arr / arr.sum(axis, keepdims=True)


def calculate_p_utterance_given_gamma(possible_utterances, possible_observations, 
                                      utterance_observation_compatibility, gamma):
    """
    The probability of an utterance *being true* (NOT being produced) given a gamma
    To calculate it:
        - Calculate the probability of each observation given the gamma
        - For each utterance, sum the probability of those observations that verify the utterance

    Parameters
    ----------
    possible_utterances, possible_observations: arrays
        See return value of get_and_clean_data
    utterance_observation_compatibility: Boolean or int 2d array
        array that says whether each observation is compatible
        with each utterance.
    gamma: float
        Binomial parameter (see model description for explanation)
    Returns
    -------
    2d array
        Array with the probability of each utterance being true
        given a gamma.
    """
    p_obs_given_gamma = (
        stats.binom.pmf(
            possible_observations, 
            n=12, 
            p=gamma
        )
        .prod(-1)[None]
    )
    return (p_obs_given_gamma * utterance_observation_compatibility).sum(1)


def calculate_prob_quant(quant, bias, n_obs):
    """
    Probability of the world producing a state that verifies the quantifier
    (marginalized across states)
    """
    if quant == 'all':
        return stats.binom.pmf(n_obs, n=n_obs, p=bias)
    elif quant == 'some':
        return stats.binom.pmf(range(1, n_obs+1), n=n_obs, p=bias).sum()
    elif quant == 'most':
        return stats.binom.pmf(range(int(np.ceil(n_obs/2)), n_obs+1), n=n_obs, p=bias).sum()
    elif quant == 'none':
        return stats.binom.pmf(0, n=n_obs, p=bias)


def run_michael_method(utterance, bias, n_students=5, n_questions=12):
    """
    Use the method to calculate the probability of an utterance
    being true given a bias (gamma)
    This is a reimplementation in python of the code in Michael Franke's
    original R code for the basic model.
    """
    if utterance[2] == 'wrong':
        bias = 1-bias
    # probability that the quantifier is true for a sample
    inner_prob = calculate_prob_quant(utterance[1], bias, n_questions)
    return calculate_prob_quant(utterance[0], inner_prob, n_students)


def calculate_argumentative_strength(possible_utterances, possible_observations, 
                                     gamma_prove, gamma_disprove, michael_method=True,
                                     michael_kwargs=dict()):
    """
    Calculate the argumentative strength of each possible utterance given each possible state
    and a gamma to prove and a gamma to disprove.
    
    The argumentative strength of an utterance given a value of gamma that one wants to prove
    and a value of gamma that one wants to disprove is equal to:
    log(p(utterance | gamma_prove)) - log(p(utterance | gamma_disprove))

    NOTE: Michael's method considers all *possible* observations 
    (i.e. every possible number of correct answers from each student)
    while my method considers just the observations that
    can be actually made in the experiment.
    """
    
    if michael_method:
        
        p_prove = [
            run_michael_method(u, gamma_prove, **michael_kwargs)
            for u in possible_utterances
        ]

        p_disprove = [
            run_michael_method(u, gamma_disprove, **michael_kwargs)
            for u in possible_utterances
        ]

        return np.log(p_prove) - np.log(p_disprove)

    else:
        utterance_observation_compatibility = np.stack([
            verify(*a, possible_observations)
            for a in possible_utterances
        ])

        return (
            np.log(calculate_p_utterance_given_gamma(
                possible_utterances, 
                possible_observations, 
                utterance_observation_compatibility, 
                gamma_prove
            )) -
            np.log(calculate_p_utterance_given_gamma(
                possible_utterances, 
                possible_observations, 
                utterance_observation_compatibility, 
                gamma_disprove
            ))
        )


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


def calculate_logp_data(data, alpha, beta, 
                        possible_observations, possible_utterances, 
                        like_cogsci_paper=False):
    """
    Calculate the logp of the data for the first version of the model in the cogsci poster.
    This is meant for initial testing with numpy to compare with original greta implementation.
    
    Parameters
    ----------
    data, possible_observations, possible_utterances: df, arrays
        See value returned by get_and_clean_data
    alpha, beta: floats
        See description of model
    like_cogsci_paper: bool
        Reproduces Michael's implementation.
    """
    
    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])
    
    if like_cogsci_paper:
        p_observation_given_utterance = 1/utterance_observation_compatibility.sum(1)[:,None]
        p_observation_given_utterance = np.tile(p_observation_given_utterance, (1, 20))
        # shape: (utterances, observations)
        truth_matrix = (1-np.stack([
            verify(*a, possible_observations)
            for a in possible_utterances
        ])) *-60
    else:
        p_observation_given_utterance = normalize(utterance_observation_compatibility,1)
        truth_matrix = 0
    
    argumentative_strengths_positive = calculate_argumentative_strength(
        possible_utterances, 
        possible_observations, 
        gamma_prove=0.85,
        gamma_disprove=0.15
    )
    
    argumentative_strengths_negative = calculate_argumentative_strength(
        possible_utterances, 
        possible_observations, 
        gamma_prove=0.15,
        gamma_disprove=0.85
    )
    
    # The probability of an observation given an utterance
    # is simply 1 divided by the number of observations compatible
    # with the utterance. 
    # (we are assuming here a uniform prior over observations)
    
    costs = get_costs(possible_utterances)
    
    p_utterance_given_observation_high = calculate_pragmatic_speaker(
        argumentative_strengths_positive,
        p_observation_given_utterance, 
        costs,
        alpha, 
        beta,
        truth_matrix
    )
        
    p_utterance_given_observation_low = calculate_pragmatic_speaker(
        argumentative_strengths_negative,
        p_observation_given_utterance,
        costs,
        alpha, 
        beta,
        truth_matrix
    )
    
    p_utterance_given_observation = np.stack((
        p_utterance_given_observation_low,
        p_utterance_given_observation_high
    ))
    
    return np.log(p_utterance_given_observation[
        data.condition,
        data.index_utterance, 
        data.index_observation
    ])

