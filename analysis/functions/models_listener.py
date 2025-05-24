import pymc3 as pm
import numpy as np
from theano import tensor as tt

from functions.helper_functions import (
    verify, 
    normalize, 
    theano_calculate_pragmatic_speaker,
    theano_normalize
)

from functions.argstrengths import (
    calculate_argumentative_strength, 
    calculate_maximin_argstrength,
    theano_calculate_pragmatic_argstrength, 
    calculate_nonparametric_argstrength
)

def factory_model_base(data, possible_observations, possible_utterances, include_observed=True):
    
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
        
        if include_observed:
            data_response = pm.Data(
                'observed', 
                data.response
            )
        else:
            data_response = None

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
        
        # normalize so each row (signal) is a prob vector
        # to get listener-side probabilities
        p_observation_given_utterance = theano_normalize(
            p_utterance_given_observation,
            axis=1
        )
        
        # shape (2, trials)
        # for each datapoint, the prob of picking 
        # left and right exam results given the observed utterance
        unnorm_p_selection = p_observation_given_utterance[
            # shown utterance
            data.index_utterance,
            # probabilities of left and right options
            [data.left, data.right]
        ]
        tt.printing.Print()(unnorm_p_selection.shape)
                
        # normalize to find probability of choosing
        # left or right image
        p_selection = theano_normalize(
            unnorm_p_selection,
            0
        )
        
        choices = pm.Categorical(
            'selection',
            p_selection.T,
            observed=data_response,
            shape=len(data)
        )
    
    return model_base

def factory_model_base_hierarchical(
        data, possible_observations, 
        possible_utterances, 
        include_observed=True):
    
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

    with pm.Model() as model_base_hierarchical:

        if include_observed:
            data_response = pm.Data(
                'observed', 
                data.response
            )
        else:
            data_response = None

        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=1,
        )

        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1,
        )

        # sample one deviation from the population-level
        # for each participant
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(data.id.max()+1,),
            # NOTE: Reduced to avoid overflow
            sigma=0.1
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
            lam=1
        )

        costs = mask_none * cost

        # dimensions (participant, utterance, observation)
        p_utterance_given_observation = theano_calculate_pragmatic_speaker(
            p_observation_given_utterance, 
            costs,
            alpha, 
        )

        p_observation_given_utterance = theano_normalize(
            p_utterance_given_observation,
            # normalize each utterance across observations
            -1
        )

        # shape (2, # trials)
        # for each datapoint, the prob of picking 
        # left and right exam results given the observed utterance
        unnorm_p_selection = p_observation_given_utterance[
            data.id,
            # shown utterance
            data.index_utterance,
            # probabilities of left and right options
            [data.left, data.right]
        ]

        # normalize to find probability of choosing
        # left or right image
        p_selection = theano_normalize(
            unnorm_p_selection,
            0
        )

        tt.printing.Print()(tt.stack([p_selection.min(), p_selection.max()]))

        choices = pm.Categorical(
            'selection',
            p_selection.T,
            observed=data_response,
            shape=len(data)
        )

        return model_base_hierarchical


def factory_model_lr_argstrength(data, possible_observations, possible_utterances, 
                                 include_observed=True):
    
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

        if include_observed:
            data_response = pm.Data(
                'observed', 
                data.response
            )
        else:
            data_response = None

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
        
        # normalize so each row (signal) is a prob vector
        # to get listener-side probabilities
        p_observation_given_utterance = theano_normalize(
            p_utterance_given_observation,
            axis=-1
        )
        
        # shape (2, trial)
        # for each datapoint, the prob of picking 
        # left option given the observed utterance
        p_selection_unnorm = p_observation_given_utterance[
            data.condition,
            data.index_utterance,
            [data.left, data.right]
        ]
        p_selection = theano_normalize(
            p_selection_unnorm,
            0
        )
        
        utterances = pm.Categorical(
            'selection',
            p_selection.T,
            observed=data_response,
            shape=len(data)
        )
    
    return model_lr_argstrength

def factory_model_lr_argstrength_hierarchical(data, possible_observations, 
                                              possible_utterances, include_observed=True):
    
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

        if include_observed:
            data_response = pm.Data(
                'observed', 
                data.response
            )
        else:
            data_response = None

        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=0.5,
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1.
        )
        
        # sample one deviation from the population-level
        # for each participant
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(data.id.max()+1,),
            # NOTE: Reduced to avoid overflow
            sigma=0.1
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
            lam=1
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
        
        p_observation_given_utterance = theano_normalize(
            p_utterance_given_observation,
            # normalize each utterance across observations
            -1
        )
        
        # shape (2, trials)
        # for each datapoint, the prob of picking 
        # left and right exam results given the observed utterance
        unnorm_p_selection = p_observation_given_utterance[
            data.condition,
            data.id,
            # shown utterance
            data.index_utterance,
            # probabilities of left and right options
            [data.left, data.right]
        ]
        
        # normalize to find probability of choosing
        # left or right image
        p_selection = theano_normalize(
            unnorm_p_selection,
            0
        )
        
        choices = pm.Categorical(
            'selection',
            p_selection.T,
            observed=data_response,
            shape=len(data)
        )
            
    return model_lr_argstrength_hierarchical

def factory_model_maximin_argstrength(data, possible_observations, 
                                      possible_utterances, include_observed=True):
    
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

        if include_observed:
            data_response = pm.Data(
                'observed', 
                data.response
            )
        else:
            data_response = None

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

        # normalize so each row (signal) is a prob vector
        # to get listener-side probabilities
        p_observation_given_utterance = theano_normalize(
            p_utterance_given_observation,
            axis=-1
        )
                
        # shape (2, trial)
        # for each datapoint, the prob of picking 
        # left option given the observed utterance
        p_selection_unnorm = p_observation_given_utterance[
            data.condition,
            data.index_utterance,
            [data.left, data.right]
        ]
        
        p_selection = theano_normalize(
            p_selection_unnorm,
            0
        )
                
        utterances = pm.Categorical(
            'selection',
            p_selection.T,
            observed=data_response,
            shape=len(data)
        )
    
    return model_maximin_argstrength

def factory_model_maximin_argstrength_hierarchical(data, possible_observations, 
                                                   possible_utterances, include_observed=True):
    
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
    
    mask_none = np.any(
        possible_utterances=='none',
        axis=1
    ).astype(int)

    with pm.Model() as model_maximin_argstrength_hierarchical:
        
        if include_observed:
            data_response = pm.Data(
                'observed', 
                data.response
            )
        else:
            data_response = None
        
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
            shape=(data.id.max()+1,),
            sigma=0.1
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
        
        p_observation_given_utterance = theano_normalize(
            p_utterance_given_observation,
            # normalize each utterance across observations
            -1
        )
        
        # shape (2, trials)
        # for each datapoint, the prob of picking 
        # left and right exam results given the observed utterance
        unnorm_p_selection = p_observation_given_utterance[
            data.condition,
            data.id,
            # shown utterance
            data.index_utterance,
            # probabilities of left and right options
            [data.left, data.right]
        ]
        
        # normalize to find probability of choosing
        # left or right image
        p_selection = theano_normalize(
            unnorm_p_selection,
            0
        )
        
        choices = pm.Categorical(
            'selection',
            p_selection.T,
            observed=data_response,
            shape=len(data)
        )
            
    return model_maximin_argstrength_hierarchical

def factory_model_prag_argstrength(data, possible_observations, 
                                   possible_utterances, include_observed=True):

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

        if include_observed:
            data_response = pm.Data(
                'observed', 
                data.response
            )
        else:
            data_response = None

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
            costs=costs,
            p_observation_given_utterance=p_observation_given_utterance
        )

        prag_argstrengths_negative = theano_calculate_pragmatic_argstrength(
            possible_utterances, 
            possible_observations,
            gamma_prove=0.15, 
            gamma_disprove=0.85,
            alpha=alpha,
            costs=costs,
            p_observation_given_utterance=p_observation_given_utterance
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

        # normalize so each row (signal) is a prob vector
        # to get listener-side probabilities
        p_observation_given_utterance = theano_normalize(
            p_utterance_given_observation,
            axis=-1
        )
        
        # shape (2, trial)
        # for each datapoint, the prob of picking 
        # left option given the observed utterance
        p_selection_unnorm = p_observation_given_utterance[
            data.condition,
            data.index_utterance,
            [data.left, data.right]
        ]
        p_selection = theano_normalize(
            p_selection_unnorm,
            0
        )
        
        utterances = pm.Categorical(
            'selection',
            p_selection.T,
            observed=data_response,
            shape=len(data)
        )
    
    return model_prag_argstrength

def factory_model_prag_argstrength_hierarchical(data, possible_observations, 
                                   possible_utterances, include_observed=True):

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
        
        if include_observed:
            data_response = pm.Data(
                'observed', 
                data.response
            )
        else:
            data_response = None

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
            shape=(data.id.max()+1,),
            sigma=0.1
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
            costs=costs,
            p_observation_given_utterance=p_observation_given_utterance
        )

        prag_argstrengths_negative = theano_calculate_pragmatic_argstrength(
            possible_utterances, 
            possible_observations,
            gamma_prove=0.15, 
            gamma_disprove=0.85,
            alpha=alpha,
            costs=costs,
            p_observation_given_utterance=p_observation_given_utterance
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
        
        p_observation_given_utterance = theano_normalize(
            p_utterance_given_observation,
            # normalize each utterance across observations
            -1
        )
        
        # shape (2, trials)
        # for each datapoint, the prob of picking 
        # left and right exam results given the observed utterance
        unnorm_p_selection = p_observation_given_utterance[
            data.condition,
            data.id,
            # shown utterance
            data.index_utterance,
            # probabilities of left and right options
            [data.left, data.right]
        ]
        
        # normalize to find probability of choosing
        # left or right image
        p_selection = theano_normalize(
            unnorm_p_selection,
            0
        )
        
        choices = pm.Categorical(
            'selection',
            p_selection.T,
            observed=data_response,
            shape=len(data)
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
        
        if include_observed:
            data_response = pm.Data(
                'observed', 
                data.response
            )
        else:
            data_response = None

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

        # normalize so each row (signal) is a prob vector
        # to get listener-side probabilities
        p_observation_given_utterance = theano_normalize(
            p_utterance_given_observation,
            axis=-1
        )
        
        # shape (2, trial)
        # for each datapoint, the prob of picking 
        # left option given the observed utterance
        p_selection_unnorm = p_observation_given_utterance[
            data.condition,
            data.index_utterance,
            [data.left, data.right]
        ]
        p_selection = theano_normalize(
            p_selection_unnorm,
            0
        )
        
        utterances = pm.Categorical(
            'selection',
            p_selection.T,
            observed=data_response,
            shape=len(data)
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

    mask_none = np.any(
        possible_utterances=='none', 
        axis=1
    ).astype(int)

    with pm.Model() as model_nonparametric_argstrength_hierarchical:

        if include_observed:
            data_response = pm.Data(
                'observed', 
                data.response
            )
        else:
            data_response = None
        
        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=1,
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1
        )
        
        # sample one deviation from the population-level
        # for each participant
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(data.id.max()+1,),
            sigma=0.1
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
        
        p_observation_given_utterance = theano_normalize(
            p_utterance_given_observation,
            # normalize each utterance across observations
            -1
        )

        # shape (2, trials)
        # for each datapoint, the prob of picking 
        # left and right exam results given the observed utterance
        unnorm_p_selection = p_observation_given_utterance[
            data.condition,
            data.id,
            # shown utterance
            data.index_utterance,
            # probabilities of left and right options
            [data.left, data.right]
        ]
        
        # normalize to find probability of choosing
        # left or right image
        p_selection = theano_normalize(
            unnorm_p_selection,
            0
        )
        
        choices = pm.Categorical(
            'selection',
            p_selection.T,
            observed=data_response,
            shape=len(data)
        )
            
    return model_nonparametric_argstrength_hierarchical