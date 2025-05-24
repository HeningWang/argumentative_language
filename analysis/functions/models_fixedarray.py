import pymc3 as pm
import numpy as np
from theano import tensor as tt

from functions.helper_functions import (
    verify, 
    normalize, 
    theano_calculate_pragmatic_speaker,
    get_costs,
    calculate_pragmatic_speaker
)

from functions.argstrengths import (
    calculate_argumentative_strength, 
    calculate_maximin_argstrength,
    theano_calculate_pragmatic_argstrength, 
    calculate_nonparametric_argstrength
)


def calculate_logp_data(data, alpha, beta, 
                        possible_observations, possible_utterances, 
                        like_cogsci_paper=False):
    """
    Calculate the logp of the data in numpy for the simplest version of the model.
    
    Parameters
    ----------
    data, possible_observations, possible_utterances: df, arrays
        See value returned by get_and_clean_data
    alpha, beta: floats
        See description of models
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



def factory_model_base(data, possible_observations, 
                       possible_utterances, include_observed=True):
    """
    Vanilla RSA with pooled alpha
    (Originally called: `_lr_argstrength_noarg` in the explorative notebook)
    """
    
    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])

    # literal listener
    p_observation_given_utterance = normalize(
        utterance_observation_compatibility,
        1
    )
                               
    mask_none = np.any(
        possible_utterances=='none', 
        axis=1
    ).astype(int)
    
    with pm.Model() as model_base:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=5
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )

        costs = mask_none * cost

        p_utterance_given_observation = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
        )

        p_production = p_utterance_given_observation[
            :, 
            data.index_observation
        ]

        pm.Categorical(
            'utterances',
            p_production.T,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_base


def factory_model_base_hierarchical(
        data, possible_observations, 
        possible_utterances, include_observed=True):
    """
    Vanilla RSA with hierarchical alpha
    (called `hierarchical_alpha_noconfusion_noarg` in explorative.ipybn)
    """
    
    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])

    # literal listener
    p_observation_given_utterance = normalize(
        utterance_observation_compatibility,
        1
    )
                               
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)
    
    with pm.Model() as model_base_hierarchical:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )
        
#         alpha_mu = pm.Normal(
#             'alpha_mu', 
#             mu=0,
#             sigma=5
#         )
#         alpha_sigma = pm.HalfNormal(
#             'alpha_sigma',
#             sigma=1
#         )
        
#         alpha = pm.TruncatedNormal(
#             'alpha',
#             mu=alpha_mu,
#             sigma=alpha_sigma,
#             lower=0,
#             # number of participants
#             shape=(data.id.max()+1,)
#         )

        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=5,
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1,
        )
        
        # sample one deviation from the population-level
        # for each participant
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(data.id.max()+1,)
        )
        
        alpha = pm.Deterministic(
            'alpha',
            pm.math.exp(
                alpha_mu + 
                alpha_sigma * alpha_zs
            )
        )
        
        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )

        costs = mask_none * cost

        # dimensions (participant, utterance, observation)
        p_utterance_given_observation = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
        )
                        
        p_production = p_utterance_given_observation[
            data['id'],
            :, 
            data['index_observation']
        ]
        
        utterances = pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None,
            shape=len(data)
        )
    
    return model_base_hierarchical


def factory_model_lr_argstrength(data, possible_observations, possible_utterances, 
                                 include_observed=True):
    """
    Log-likelihood ratio argstrength RSA
    Completely pooled $\alpha$ and $\beta$
    (called `_no_confusion` in explorative.ipybn)
    """
    
    argumentative_strengths_positive = calculate_argumentative_strength(
        possible_utterances, 
        possible_observations, 
        gamma_prove=0.85, 
        gamma_disprove=0.15,
    )
    argumentative_strengths_negative = calculate_argumentative_strength(
        possible_utterances, 
        possible_observations, 
        gamma_prove=0.15,
        gamma_disprove=0.85,
    )

    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])

    # literal listener
    p_observation_given_utterance = normalize(utterance_observation_compatibility,1)
                               
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)
    
    with pm.Model() as model_lr_argstrength:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=5
        )

        beta = pm.Uniform(
            'beta',
            lower=0,
            upper=1
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )

        costs = mask_none * cost

        p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            argumentative_strengths_positive
        )

        p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            argumentative_strengths_negative
        )

        p_utterance_given_observation = tt.stack((
            p_utterance_given_observation_low,
            p_utterance_given_observation_high
        ))

        p_production = p_utterance_given_observation[
            data.condition,
            :, 
            data.index_observation
        ]

        pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_lr_argstrength


def factory_model_lr_argstrength_hierarchical(data, possible_observations, 
                                              possible_utterances, include_observed=True):
    """
    Log-likelihood ratio argstrength RSA
    By-participant $\alpha$, $\beta$
    """
    argumentative_strengths_positive = calculate_argumentative_strength(
        possible_utterances, 
        possible_observations, 
        gamma_prove=0.85, 
        gamma_disprove=0.15,
    )
    argumentative_strengths_negative = calculate_argumentative_strength(
        possible_utterances, 
        possible_observations, 
        gamma_prove=0.15,
        gamma_disprove=0.85,
    )

    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])

    # literal listener
    p_observation_given_utterance = normalize(
        utterance_observation_compatibility,
        1
    )

    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)

    with pm.Model() as model_lr_argstrength_hierarchical:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=5,
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1
        )
        
        # sample one deviation from the population-level
        # for each participant
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(data.id.max()+1,)
        )
        
        alpha = pm.Deterministic(
            'alpha',
            pm.math.exp(
                alpha_mu + 
                alpha_sigma * alpha_zs
            )
        )

        # sample the hyperparameters
        # for the beta parameter
        beta_mu = pm.Normal(
            'beta_mu', 
            mu=0,
            sigma=1
        )
        beta_sigma = pm.HalfNormal(
            'beta_sigma',
            sigma=1
        ) 

        # condition_confusion_participant has length (# participants)
        # and contains the confusion probabilities particpant-wise
        beta_offset = pm.Normal(
            'beta_offset',
            # number of participants
            shape=(data.id.max()+1,)
        )
        beta = pm.Deterministic(
            'beta',
            pm.math.invlogit(beta_mu + beta_offset * beta_sigma)
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )
        
        costs = mask_none * cost

        # dims (participant, utterances, observations)
        p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            argumentative_strengths_positive
        )
        
        # dims (participant, utterances, observations)
        p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            argumentative_strengths_negative
        )
        
        # dims (condition, participant, utterance, observation)
        p_utterance_given_observation = tt.stack((
            p_utterance_given_observation_low,
            p_utterance_given_observation_high
        ))

        # dims (datapoint, utterance)
        p_production = p_utterance_given_observation[
            data['condition'],
            data['id'],
            :,
            data['index_observation']
        ]
        
        pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None
        )
            
    return model_lr_argstrength_hierarchical


def factory_model_maximin_argstrength(data, possible_observations, 
                                      possible_utterances, include_observed=True):
    """
    Maximin argstrength RSA
    Completely pooled $\alpha$ and $\beta$
    """
    
    maximin_argstrengths_positive = calculate_maximin_argstrength(
        possible_utterances, 
        possible_observations, 
        gamma_prove=0.85, 
        gamma_disprove=0.15
    )
    maximin_argstrengths_negative = calculate_maximin_argstrength(
        possible_utterances, 
        possible_observations, 
        gamma_prove=0.15,
        gamma_disprove=0.85
    )

    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])

    # literal listener
    p_observation_given_utterance = normalize(utterance_observation_compatibility,1)
                               
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)
    
    with pm.Model() as model_maximin_argstrength:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=5
        )

        beta = pm.Uniform(
            'beta',
            lower=0,
            upper=1
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )

        costs = mask_none * cost
        
        p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            maximin_argstrengths_positive
        )

        p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            maximin_argstrengths_negative
        )

        p_utterance_given_observation = tt.stack((
            p_utterance_given_observation_low,
            p_utterance_given_observation_high
        ))

        p_production = p_utterance_given_observation[
            data.condition,
            :, 
            data.index_observation
        ]

        pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_maximin_argstrength


def factory_model_maximin_argstrength_hierarchical(data, possible_observations, 
                                                   possible_utterances, include_observed=True):
    """
    Maximin argstrength RSA
    By-participant $\alpha$, $\beta$
    """
    
    argumentative_strengths_positive = calculate_maximin_argstrength(
        possible_utterances, 
        possible_observations, 
        gamma_prove=0.85, 
        gamma_disprove=0.15
    )
    
    argumentative_strengths_negative = calculate_maximin_argstrength(
        possible_utterances, 
        possible_observations, 
        gamma_prove=0.15, 
        gamma_disprove=0.85
    )

    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])

    # literal listener
    p_observation_given_utterance = normalize(
        utterance_observation_compatibility,
        1
    )
    
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)

    with pm.Model() as model_maximin_argstrength_hierarchical:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )
        
        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=0.5,
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1
        )
        
        # sample one deviation from the population-level
        # for each participant
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(data.id.max()+1,)
        )
        
        alpha = pm.Deterministic(
            'alpha',
            pm.math.exp(
                alpha_mu + 
                alpha_sigma * alpha_zs
            )
        )

        # sample the hyperparameters
        # for the beta parameter
        beta_mu = pm.Normal(
            'beta_mu', 
            mu=0,
            sigma=1
        )
        beta_sigma = pm.HalfNormal(
            'beta_sigma',
            sigma=1
        ) 

        # condition_confusion_participant has length (# participants)
        # and contains the confusion probabilities particpant-wise
        beta_offset = pm.Normal(
            'beta_offset',
            # number of participants
            shape=(data.id.max()+1,)
        )
        beta = pm.Deterministic(
            'beta',
            pm.math.invlogit(
                beta_mu + 
                beta_offset * beta_sigma
            )
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )

        costs = mask_none * cost

        # dims (participant, utterances, observations)
        p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            argumentative_strengths_positive
        )
        
        # dims (participant, utterances, observations)
        p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            argumentative_strengths_negative
        )
        
        # dims (condition, participant, utterance, observation)
        p_utterance_given_observation = tt.stack((
            p_utterance_given_observation_low,
            p_utterance_given_observation_high
        ))

        # dims (datapoint, utterance)
        p_production = p_utterance_given_observation[
            data['condition'],
            data['id'],
            :,
            data['index_observation']
        ]
        
        pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None
        )
            
    return model_maximin_argstrength_hierarchical


def factory_model_prag_argstrength(data, possible_observations, 
                                   possible_utterances, include_observed=True):
    """
    $S_1$ argstrength RSA
    Completely pooled $\alpha$ and $\beta$
    """

    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)

    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])

    # literal listener
    p_observation_given_utterance = normalize(
        utterance_observation_compatibility,
        1
    )

    with pm.Model() as model_prag_argstrength:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=5
        )

        beta = pm.Uniform(
            'beta',
            lower=0,
            upper=1
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )

        costs = mask_none * cost

        prag_argstrengths_positive = theano_calculate_pragmatic_argstrength(
            possible_utterances, 
            possible_observations,
            gamma_prove=0.85, 
            gamma_disprove=0.15,
            alpha=alpha,
            costs=costs
        )

        prag_argstrengths_negative = theano_calculate_pragmatic_argstrength(
            possible_utterances, 
            possible_observations,
            gamma_prove=0.15, 
            gamma_disprove=0.85,
            alpha=alpha,
            costs=costs
        )

        p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            prag_argstrengths_positive
        )

        p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            prag_argstrengths_negative
        )

        p_utterance_given_observation = tt.stack((
            p_utterance_given_observation_low,
            p_utterance_given_observation_high
        ))

        p_production = p_utterance_given_observation[
            data['condition'],
            :,
            data['index_observation']
        ]

        pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None
        )
    
    return model_prag_argstrength


def factory_model_prag_argstrength_hierarchical(data, possible_observations, 
                                   possible_utterances, include_observed=True):
    """
    $S_1$ argstrength RSA
    By-participant $\alpha$ and $\beta$
    """

    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)

    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])

    # literal listener
    p_observation_given_utterance = normalize(
        utterance_observation_compatibility,
        1
    )

    with pm.Model() as model_prag_argstrength_hierarchical:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=5,
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1
        )
        
        # sample one deviation from the population-level
        # for each participant
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(data.id.max()+1,)
        )
        
        alpha = pm.Deterministic(
            'alpha',
            pm.math.exp(
                alpha_mu + 
                alpha_sigma * alpha_zs
            )
        )

        # sample the hyperparameters
        # for the beta parameter
        beta_mu = pm.Normal(
            'beta_mu', 
            mu=0,
            sigma=1
        )
        beta_sigma = pm.HalfNormal(
            'beta_sigma',
            sigma=1
        ) 

        # condition_confusion_participant has length (# participants)
        # and contains the confusion probabilities particpant-wise
        beta_offset = pm.Normal(
            'beta_offset',
            # number of participants
            shape=(data.id.max()+1,)
        )
        beta = pm.Deterministic(
            'beta',
            pm.math.invlogit(
                beta_mu + 
                beta_offset * beta_sigma
            )
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )

        costs = mask_none * cost

        prag_argstrengths_positive = theano_calculate_pragmatic_argstrength(
            possible_utterances, 
            possible_observations,
            gamma_prove=0.85, 
            gamma_disprove=0.15,
            alpha=alpha,
            costs=costs
        )

        prag_argstrengths_negative = theano_calculate_pragmatic_argstrength(
            possible_utterances, 
            possible_observations,
            gamma_prove=0.15, 
            gamma_disprove=0.85,
            alpha=alpha,
            costs=costs
        )
                
        p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            prag_argstrengths_positive
        )

        p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            prag_argstrengths_negative
        )

        p_utterance_given_observation = tt.stack((
            p_utterance_given_observation_low,
            p_utterance_given_observation_high
        ))
        
        p_production = p_utterance_given_observation[
            data['condition'],
            data['id'],
            :,
            data['index_observation']
        ]

        pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None
        )
    
    return model_prag_argstrength_hierarchical


def factory_model_nonparametric_argstrength(data, possible_observations, possible_utterances, 
                                            include_observed=True):
    
    argumentative_strengths_positive = calculate_nonparametric_argstrength(
        possible_utterances, 
        possible_observations, 
        condition='high'
    ).flatten()
    argumentative_strengths_negative = 1-argumentative_strengths_positive

    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])

    # literal listener
    p_observation_given_utterance = normalize(
        utterance_observation_compatibility,
        1
    )
                               
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)
    
    with pm.Model() as model_nonparametric_argstrength:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=5
        )

        beta = pm.Uniform(
            'beta',
            lower=0,
            upper=1
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )

        costs = mask_none * cost

        p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            argumentative_strengths_positive
        )

        p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            argumentative_strengths_negative
        )

        p_utterance_given_observation = tt.stack((
            p_utterance_given_observation_low,
            p_utterance_given_observation_high
        ))

        p_production = p_utterance_given_observation[
            data.condition,
            :, 
            data.index_observation
        ]

        utterances = pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_nonparametric_argstrength


def factory_model_nonparametric_argstrength_hierarchical(data, possible_observations, 
                                                         possible_utterances, include_observed=True):
    
    argumentative_strengths_positive = calculate_nonparametric_argstrength(
        possible_utterances, 
        possible_observations, 
        condition='high'
    ).flatten()
    
    argumentative_strengths_negative = 1-argumentative_strengths_positive

    utterance_observation_compatibility = np.stack([
        verify(*a, possible_observations)
        for a in possible_utterances
    ])

    # literal listener
    p_observation_given_utterance = normalize(
        utterance_observation_compatibility,
        1
    )

    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)

    with pm.Model() as model_nonparametric_argstrength_hierarchical:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )
        
        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=5,
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1
        )
        
        # sample one deviation from the population-level
        # for each participant
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(data.id.max()+1,)
        )
        
        alpha = pm.Deterministic(
            'alpha',
            pm.math.exp(
                alpha_mu + 
                alpha_sigma * alpha_zs
            )
        )

        # sample the hyperparameters
        # for the beta parameter
        beta_mu = pm.Normal(
            'beta_mu', 
            mu=0,
            sigma=1
        )
        beta_sigma = pm.HalfNormal(
            'beta_sigma',
            sigma=1
        ) 

        # condition_confusion_participant has length (# participants)
        # and contains the confusion probabilities particpant-wise
        beta_offset = pm.Normal(
            'beta_offset',
            # number of participants
            shape=(data.id.max()+1,)
        )
        beta = pm.Deterministic(
            'beta',
            pm.math.invlogit(beta_mu + beta_offset * beta_sigma)
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )
        
        costs = mask_none * cost

        # dims (participant, utterances, observations)
        p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            argumentative_strengths_positive
        )
        
        # dims (participant, utterances, observations)
        p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
            beta,
            argumentative_strengths_negative
        )
        
        # dims (condition, participant, utterance, observation)
        p_utterance_given_observation = tt.stack((
            p_utterance_given_observation_low,
            p_utterance_given_observation_high
        ))

        # dims (datapoint, utterance)
        p_production = p_utterance_given_observation[
            data['condition'],
            data['id'],
            :,
            data['index_observation']
        ]
        
        pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None
        )
            
    return model_nonparametric_argstrength_hierarchical