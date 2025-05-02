import numpy as np
from scipy import stats
from theano import tensor as tt

from functions.helper_functions import (
    verify, normalize, theano_calculate_pragmatic_speaker
)


def calculate_argumentative_strength(possible_utterances, possible_observations, 
                                     gamma_prove, gamma_disprove):
    """
    Calculate the argumentative strength of each possible utterance given each possible state
    and a gamma to prove and a gamma to disprove.
    
    The argumentative strength of an utterance given a value of gamma that one wants to prove
    and a value of gamma that one wants to disprove is equal to:
    log(p(utterance | gamma_prove)) - log(p(utterance | gamma_disprove))
    """

    def calculate_p_utterance_given_gamma(possible_observations, 
                                          utterance_observation_compatibility, gamma):
        """
        The probability of an utterance *being true* (NOT being produced) given a gamma
        To calculate it:
            - Calculate the probability of each observation given the gamma
            - For each utterance, sum the probability of those observations that verify the utterance

        Parameters
        ----------
        utterance_observation_compatibility: Boolean or int 2d array
            array that says whether each observation is compatible
            with each utterance.
        gamma: float
            Binomial parameter (see model description for explanation)
        Returns
        -------
        array
            Array with the probability of each utterance being true
            given a gamma.
        """
        # calculates the probability of each observation given the gamma.
        # Dims (observation)
        p_obs_given_gamma = (
            stats.binom.pmf(
                possible_observations, 
                n=possible_observations.max(), 
                p=gamma
            )
            .prod(-1)[None]
        )

        # since exactly one observation has to be true,
        # normalize across observations
        p_obs_given_gamma = normalize(p_obs_given_gamma)

        # shape (utterance)
        return (p_obs_given_gamma * utterance_observation_compatibility).sum(1)
    
    utterance_observation_compatibility = np.stack([
        verify(
            *a, 
            possible_observations,
            n_answers=possible_observations.max(),
            n_students=possible_observations.shape[1]            
        )
        for a in possible_utterances
    ])

    return (
        np.log(
            calculate_p_utterance_given_gamma(
                possible_observations, 
                utterance_observation_compatibility, 
                gamma_prove
            )
        ) -
        np.log(
            calculate_p_utterance_given_gamma(
                possible_observations, 
                utterance_observation_compatibility, 
                gamma_disprove
            )
        )
    )


def calculate_maximin_argstrength(possible_utterances, possible_observations, 
                                  gamma_prove, gamma_disprove):
    
    def calculate_p_obs_given_utterance_and_gamma_maximin(possible_observations, 
                                              utterance_observation_compatibility,
                                              gamma):
        """
        Probability of each observation given the gamma parameter
        among the observations compatible with the utterance
        """
        p_obs_given_gamma = (
            # probability of each individual student answer
            stats.binom.pmf(
                possible_observations, 
                n=possible_observations.max(), 
                p=gamma
            )
            # probability of that combination of answers
            .prod(-1)[None]
        )
        p_obs_given_gamma = normalize(p_obs_given_gamma)
        # for each signal (row), only consider
        # the observations compatible with that signal,
        # else set 0
        return p_obs_given_gamma * utterance_observation_compatibility
    
    utterance_observation_compatibility = np.stack([
        verify(
            *a, 
            possible_observations,
            n_answers=possible_observations.max(),
            n_students=possible_observations.shape[1]
        )
        for a in possible_utterances
    ])
    
    # set -inf where incompatible and 1 where compatible
    utterance_observation_compatibility = np.where(
        utterance_observation_compatibility == 0,
        np.inf,
        utterance_observation_compatibility
    )
    
    # dimensions (utterances, observations)
    logp_for = np.log(
        calculate_p_obs_given_utterance_and_gamma_maximin(
            possible_observations, 
            utterance_observation_compatibility, 
            gamma_prove
        )
    )
    
    logp_against = np.log(
        calculate_p_obs_given_utterance_and_gamma_maximin(
            possible_observations, 
            utterance_observation_compatibility, 
            gamma_disprove
        )
    )
    
    return np.nanmin(logp_for-logp_against, 1)


def theano_calculate_pragmatic_argstrength(possible_utterances, possible_observations,
                                           gamma_prove, gamma_disprove, alpha, costs):
    """
    Parameters
    ----------
    alpha: theano float or 1-d tensor
        if single value, that means the model is completely pooled
        if a tensor, there is one value per participant and model is 
        pooled by-participant.
    """
    
    def calculate_p_observation_given_gamma(possible_observations, gamma):
        p_obs_given_gamma = (
            stats.binom.pmf(
                possible_observations,
                n=possible_observations.max(),
                p=gamma
            )
            .prod(-1)[None]
        )
        # normalize across utterances
        return p_obs_given_gamma / p_obs_given_gamma.sum()
    
    p_observation_given_utterance = normalize(np.stack([
        verify(
            *a, 
            possible_observations,
            n_answers=possible_observations.max(),
            n_students=possible_observations.shape[1]
        )
        for a in possible_utterances
    ]), 1)
    
    # probability of each observation given gammas
    # shape (utterance, observation)
    p_obs_prove = calculate_p_observation_given_gamma(
        possible_observations, 
        gamma_prove
    )
    p_obs_disprove = calculate_p_observation_given_gamma(
        possible_observations, 
        gamma_disprove
    )
    
    hierarchical = alpha.type.ndim > 0
    if hierarchical:
        # add participant dimension
        p_obs_prove = p_obs_prove[None]
        p_obs_disprove = p_obs_disprove[None]
    
    # pragmatic speaker who assumes an L0
    # with a uniform prior over states.
    # Dimensions are either:
    # (utterance, observation) or 
    # (participant, utterance, observation)
    pragmatic_speaker = theano_calculate_pragmatic_speaker(
        p_observation_given_utterance,  
        costs,
        alpha
    )
    
    # sum across observations
    return (
        tt.log((pragmatic_speaker * p_obs_prove).sum(-1)) -
        tt.log((pragmatic_speaker * p_obs_disprove).sum(-1))
    )


def calculate_nonparametric_argstrength(possible_utterances, possible_observations, 
                                        condition):
    """
    Calculate a non parametric version of the argumentative strength 
    of each possible utterance given each possible state.
    """
    assert condition in ['high', 'low'], 'Condition not known'
    
    # (signal, state)
    utterance_observation_compatibility = np.stack([
        verify(
            *a, 
            possible_observations,
            n_answers=possible_observations.max(),
            n_students=possible_observations.shape[1]
        )
        for a in possible_utterances
    ])
    
    # (signal, state)
    # the .sum(1) gives the total number of correct answers across all students
    # for that exam
    helparr = utterance_observation_compatibility * possible_observations.sum(1)
    
    # (signal)
    argstrength = (
        # the .sum(1) gives the total number of correct
        # across states compatible with the signal
        helparr.sum(1, keepdims=True)
        # take mean across signals
        / utterance_observation_compatibility.sum(1, keepdims=True)
    )
    
    # center around 0
    argstrength = argstrength - argstrength.mean()
    
    if condition == 'low':
        argstrength = -argstrength

    return argstrength