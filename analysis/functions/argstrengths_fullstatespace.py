from collections import Counter
from itertools import combinations_with_replacement, product
from math import lgamma, factorial
from os.path import join
import numpy as np
from scipy import stats
from scipy.special import binom

from functions.helper_functions import verify, load_argstrengths


def calculate_argumentative_strength_fullstatespace(
        possible_utterances, possible_observations, 
        gamma_prove, gamma_disprove, savefolder=None):
    
    """
    Arguments
    ---------
    possible_observations: array
        This is only used to get the number of answers and of students
        but the actual argstrength is calculated on all the possible states.

    Returns
    -------
    array
        Array with the argumentative strength of each possible utterance
        shape (n_utterances,)
    """
    
    def multinomial_coefficient(state):
        """
        Calculates the multinomial coefficient for a given unordered state.

        Args:
            state (tuple): Unordered state as a tuple of values.

        Returns:
            int: Number of ordered states corresponding to the unordered state.
        """
        counts = Counter(state).values()
        numerator = factorial(len(state))
        denominator = 1
        for count in counts:
            denominator *= factorial(count)
        return numerator // denominator  # Use integer division

    def calculate_p_utterance_given_gamma(n_answers, n_students, gamma, possible_utterances):
        iterator = combinations_with_replacement(range(n_answers+1), n_students)
        utterance_probs = np.zeros(len(possible_utterances))
        for i,k in enumerate(iterator):
            k = np.array(k)
            j = multinomial_coefficient(k)
            utterance_observation_compatibility = np.stack([
                verify(*a, k, n_answers, n_students)
                for a in possible_utterances
            ]).flatten()
            probs = (
                j 
                * gamma**k.sum() * (1-gamma)**(n_answers-k).sum() 
                * binom(n_answers, k).prod()
            )
            utterance_probs += probs*utterance_observation_compatibility
        return utterance_probs
        
    n_answers = possible_observations.max()
    n_students = possible_observations.shape[1]
    
    savename_dict = {
        "kind": "lr",
        "nanswers": n_answers,
        "nstudents": n_students,
        "gammaprove": gamma_prove,
        "gammadisprove": gamma_disprove,
    }
    
    if savefolder is not None:
        savename = join(
            savefolder, 
            "-".join([
                f"{key}={value}" 
                for key, value 
                in savename_dict.items()
            ])+'_fullstatespace.npy'
        )
    else:
        savename = None
    
    if (argstrengths := load_argstrengths(savename)) is not None:
        
        return argstrengths
    
    logliks_prove = np.log(calculate_p_utterance_given_gamma(
        n_answers, 
        n_students, 
        gamma_prove, 
        possible_utterances
    ))

    logliks_disprove = np.log(calculate_p_utterance_given_gamma(
        n_answers, 
        n_students, 
        gamma_disprove, 
        possible_utterances
    ))

    argstrengths = logliks_prove - logliks_disprove

    if savename is not None:
        with open(savename, 'wb') as openf:
            np.save(openf, argstrengths)

    return argstrengths

    
def calculate_argumentative_strength_slow_fullstatespace(possible_utterances, possible_observations, 
                                     gamma_prove, gamma_disprove):
    """
    Calculate the argumentative strength of each possible utterance given each possible state
    and a gamma to prove and a gamma to disprove.
    
    Note: We are calculating the argstrength based on all the possible observations,
    even the ones not in the experiment.
    
    The argumentative strength of an utterance given a value of gamma that one wants to prove
    and a value of gamma that one wants to disprove is equal to:
    log(p(utterance | gamma_prove)) - log(p(utterance | gamma_disprove))
    
    Parameters
    ----------
    possible_utterances, possible_observations: arrays
        See return value of get_and_clean_data
    utterance_observation_compatibility: Boolean or int 2d array
        array that says whether each observation is compatible
        with each utterance.
    gamma_prove, gamma_disprove: float
        Binomial parameter (see model description for explanation)
    
    Returns
    -------
    array
        Array with the argumentative strength of each possible utterance
        shape (n_utterances,)
    """
    
    def calculate_p_utterance_given_gamma(utterance_observation_compatibility, gamma):
        """
        The probability of an utterance *being true* (NOT being produced) given a gamma
        To calculate it:
            - Calculate the probability of each observation given the gamma
            - For each utterance, sum the probability of those observations that verify the utterance

        Returns
        -------
        array
            Array with the probability of each utterance being true
            given a gamma.
        """

        # calculates the probability of each observation given the gamma.
        # Dims (observation)
        p_obs_given_gamma = (
            gamma**poss_os.sum(1) 
            * (1-gamma)**(n_answers-poss_os).sum(1) 
            * binom(n_answers, poss_os).prod(1)
        )

        # shape (utterance)
        # The probability that the utterance is true given the gamma,
        # i.e., that any of the states that verify the utterance are true
        return p_obs_given_gamma @ utterance_observation_compatibility.T
    
    n_answers = possible_observations.max()
    n_students = possible_observations.shape[1]
    
    # calculate all possible states
    poss_os = np.array(list(product(range(n_answers+1), repeat=n_students)))
        
    utterance_observation_compatibility = np.stack([
        verify(
            *a, 
            poss_os,
            n_answers=n_answers,
            n_students=n_students
        )
        for a in possible_utterances
    ])
            
    log_for = np.log(calculate_p_utterance_given_gamma(
        utterance_observation_compatibility, 
        gamma_prove
    ))
    
    log_against = np.log(calculate_p_utterance_given_gamma(
        utterance_observation_compatibility, 
        gamma_disprove
    ))

    return log_for - log_against


def calculate_maximin_argstrength_fullstatespace(possible_utterances,
                                  possible_observations,
                                  gamma_prove, gamma_disprove,
                                  savefolder=None):
    """
    Maximin argumentative strength (unnormalised version) *with caching*.

        argstrength(u) =
            min_s [ log P(s | γ_prove) − log P(s | γ_disprove) ]
            over states s compatible with utterance u.
    """

    def _log_multinomial_coeff(state):
        """log (N! / ∏ n_c!) for an unordered multiset `state`."""
        return (lgamma(len(state) + 1)
                - sum(lgamma(c + 1) for c in Counter(state).values()))


    def _log_p_state_given_gamma(state, n_answers, gamma):
        """log P(state | γ) assuming exchangeable students & binomial mastery."""
        return (_log_multinomial_coeff(state)
                + sum(stats.binom.logpmf(k_i, n_answers, gamma) for k_i in state))
    
    n_answers  = int(possible_observations.max())
    n_students = possible_observations.shape[1]

    savename = None
    if savefolder is not None:
        savename = join(
            savefolder,
            "-".join([
                "kind=maximin_raw",
                f"nanswers={n_answers}",
                f"nstudents={n_students}",
                f"gammaprove={gamma_prove}",
                f"gammadisprove={gamma_disprove}",
            ]) + "_fullstatespace.npy"
        )
        if (cached := load_argstrengths(savename)) is not None:
            return cached

    min_log_ratio = np.full(len(possible_utterances), np.inf)

    for state in combinations_with_replacement(range(n_answers + 1), n_students):
        lr_state = (_log_p_state_given_gamma(state, n_answers, gamma_prove)
                    - _log_p_state_given_gamma(state, n_answers, gamma_disprove))

        compat = np.fromiter(
            (verify(*u, state, n_answers, n_students)
             for u in possible_utterances),
            dtype=bool,
            count=len(possible_utterances)
        )
        if compat.any():
            min_log_ratio[compat] = np.minimum(min_log_ratio[compat], lr_state)

    if savename is not None:
        with open(savename, "wb") as f:
            np.save(f, min_log_ratio)

    return min_log_ratio


def calculate_maximin_argstrength_slow_fullstatespace(possible_utterances,
                                  possible_observations,
                                  gamma_prove, gamma_disprove):
    """
    Maximin argumentative strength:
        min_obs log P(obs | u, γ_prove) - log P(obs | u, γ_disprove)
    """

    def calculate_p_obs_given_utterance_and_gamma_maximin(
            possible_observations,
            utterance_observation_compatibility,
            gamma):
        """
        Row‑wise posterior P(obs | u, γ):
          • compute the prior P(obs | γ)
          • zero out observations that are incompatible with u
          • renormalise *after* masking
        """
        n_answers = possible_observations.max()
        # prior: probability of each ordered classroom state
        # shape (1, n_obs)
        p_obs_given_gamma = stats.binom.pmf(
            possible_observations,
            n=n_answers,
            p=gamma
        ).prod(-1)[None]
        p_obs_given_gamma = p_obs_given_gamma * utterance_observation_compatibility
        return p_obs_given_gamma
    
    n_answers  = possible_observations.max()
    n_students = possible_observations.shape[1]
    
    poss_os = np.array(list(product(range(n_answers+1), repeat=n_students)))
    
    # compatibility matrix: 1 if (utterance, observation) is possible, else 0
    utterance_observation_compatibility = np.stack([
        verify(
            *u,
            poss_os,
            n_answers=n_answers,
            n_students=n_students
        )
        for u in possible_utterances
    ]).astype(float)

    # posterior over observations for each γ
    post_for = calculate_p_obs_given_utterance_and_gamma_maximin(
        poss_os,
        utterance_observation_compatibility,
        gamma_prove
    )
    post_against = calculate_p_obs_given_utterance_and_gamma_maximin(
        poss_os,
        utterance_observation_compatibility,
        gamma_disprove
    )
    
    # log‑ratio; incompatible cells are 0 so log→‑inf, ignored by nanmin
    with np.errstate(divide='ignore'):
        log_ratio = np.log(post_for) - np.log(post_against)

    return np.nanmin(log_ratio, axis=1)


def calculate_nonparametric_argstrength_fullstatespace(
        possible_utterances, possible_observations, condition, savefolder=None):
    
    assert condition in {"high", "low"}
    n_answers  = int(possible_observations.max())
    n_students = possible_observations.shape[1]
    
    def multinomial_coeff(state):
        """Number of permutations of an unordered state."""
        denom = 1
        for c in Counter(state).values():
            denom *= factorial(c)
        return factorial(len(state)) // denom
    
    savename = None
    if savefolder is not None:
        savename = join(
            savefolder,
            "-".join([
                "kind=nonparametric_raw",
                f"nanswers={n_answers}",
                f"nstudents={n_students}",
                f"condition={condition}",
            ]) + "_fullstatespace.npy"
        )
        if (cached := load_argstrengths(savename)) is not None:
            return cached

    # accumulators
    numer = np.zeros(len(possible_utterances), dtype=float)
    denom = np.zeros(len(possible_utterances), dtype=float)

    for state in combinations_with_replacement(range(n_answers + 1), n_students):
        # how many ordered states it stands for
        m = multinomial_coeff(state)   
        # total number of correct answers across students
        total = sum(state)
        # compatibility matrix: 1 if (utterance, observation) is possible, else 0
        compat = np.fromiter(
            (verify(*u, state, n_answers, n_students)
             for u in possible_utterances),
            dtype=bool,
            count=len(possible_utterances)
        )
        if not compat.any():
            continue
        numer[compat] += m * total
        denom[compat] += m
    # average total success per utterance
    argstrength = numer / denom
    # centre on 0
    argstrength -= argstrength.mean()
    if condition == "low":
        argstrength = -argstrength
    
    with open(savename, "wb") as f:
        np.save(f, argstrength)

    return argstrength


def calculate_nonparametric_argstrength_slow_fullstatespace(
        possible_utterances, possible_observations, condition):
    """
    Calculate a non parametric version of the argumentative strength 
    of each possible utterance given each possible state.
    """
    assert condition in ['high', 'low'], 'Condition not known'
    
    n_answers = possible_observations.max()
    n_students = possible_observations.shape[1]
    
    # calculate all possible states
    poss_os = np.array(list(product(range(n_answers+1), repeat=n_students)))
    
    utterance_observation_compatibility = np.stack([
        verify(
            *a, 
            poss_os,
            n_answers=n_answers,
            n_students=n_students
        )
        for a in possible_utterances
    ])
    
    helparr = utterance_observation_compatibility * poss_os.sum(1)
    argstrength = (
        helparr.sum(1, keepdims=True) 
        / utterance_observation_compatibility.sum(1, keepdims=True)
    )
     
    # center around 0
    argstrength = argstrength - argstrength.mean()
    
    if condition == 'low':
        argstrength = -argstrength

    return argstrength
