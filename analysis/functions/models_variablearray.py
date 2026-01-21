import numpy as np

import pymc3 as pm
import theano as T
import theano.tensor as tt

from functions.helper_functions import (
    verify,
    normalize,
    theano_calculate_pragmatic_speaker
)

from functions.argstrengths import (
    calculate_argumentative_strength,
    calculate_maximin_argstrength,
    theano_calculate_pragmatic_argstrength,
    calculate_nonparametric_argstrength
)


###### Base RSA

def factory_model_base(data, list_possible_observations, 
                       possible_utterances, include_observed=True,
                       include_S1=False):
    """
    
    The only change compared to the model with fixed array sizes is that now there's different sets of possible
    observations depending on the condition. The different array sizes might come with sets of observations 
    with different sizes, and therefore I cannot just add one more dimension to the production prob array 
    that identifies the array-size.

    Therefore I need to calculate 4 different utterance_observation_compatibility arrays and 4 different 
    argumentative strengths (in the models that have it).

    There's many ways of changing to code to adapt it for that, but the easiest is to just keep 
    the internals and create four different independent arrays for everything.
    
    Note: 
        I assume that list_possible_observations contains the observations
        in the same order in which they are indexed in 'data'!
        So for instance if data.index_observation is 0 and data.size_array_condition is 1
        then the observation has to be the same as list_possible_observation[1][0]
    
    Parameters
    ----------
    data: pd.DataFrame
        Experimental data in long format. 
        One column indicates the array-size condition
        and another column its _index_ 
    list_possible_observations: list of arrays
        A list of arrays, corresponding to the array-size conditions.
    """

    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)
    
    with pm.Model() as model_base:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=5,
            # one alpha per size array condition
            shape=len(list_possible_observations)
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )

        costs = mask_none * cost
        
        # define array to accumulate production probabilities
        p_production = tt.ones((
            len(possible_utterances), 
            len(data)
        ))
        
        # NOTE: can't assume that all array size conditions
        # have the same size in the 'observation' dimension,
        # and moreover the meaning of the 'observation' index
        # changes for each of the 4 tensors.
        # This is why I'm using this really awkward loop rather than
        # something more efficient / vectorized / theano.scan
        for i, possible_observations in enumerate(list_possible_observations):
            
            submask = (data['index_array_size_condition']==i).values
            
            p_observation_given_utterance = normalize(np.stack([
                verify(
                    *a, 
                    possible_observations,
                    n_answers=possible_observations.max(),
                    n_students=possible_observations.shape[1]
                )
                for a in possible_utterances
            ]), 1)
            
            # dimensions (utterance, observation)
            # which store the production probabilities 
            # in that size condition
            p_utterance_given_observation = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance,
                costs,
                alpha[i]
            )
            
            production_probs_masked = p_utterance_given_observation[
                    # consider all utterances
                    :,
                    # set them to the corresponding 
                    # production probabilities
                    data.loc[submask,'index_observation']
                ]
            
            p_production = tt.set_subtensor(
                p_production[
                    # all utterances
                    :,
                    # only the datapoints 
                    # in the condition of interest
                    submask
                ],
                production_probs_masked
            )

            if include_S1:
                pm.Deterministic(
                    f'S1_{i}',
                    p_utterance_given_observation
                )
        
        pm.Categorical(
            'utterances',
            p_production.T,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_base


def factory_model_base_hierarchical(data, list_possible_observations, 
                                    possible_utterances, include_observed=True):
                               
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)
    
    with pm.Model() as model_base_hierarchical:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )
        
        # Sample one population-level distribution over alpha
        # per size array condition
        
        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=1,
            # one alpha per size array condition
            shape=(len(list_possible_observations),1)
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1,
            # one alpha per size array condition
            shape=(len(list_possible_observations),1)
        )
        
        # sample one deviation from the population-level
        # for each participant
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(1,data.id.max()+1)
        )
        
        # shape (array size condition, participant)
        alpha = pm.Deterministic(
            f'alpha',
            pm.math.invlogit(
                alpha_mu + 
                alpha_sigma * alpha_zs
            ) * 5
        )
                
        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )

        costs = mask_none * cost
        
        # define array to accumulate production probabilities
        # dimensions: (datapoint, utterance)
        p_production = tt.ones((
            len(data),
            len(possible_utterances)
        ))

        for i, possible_observations in enumerate(list_possible_observations):
            
            submask = (data['index_array_size_condition']==i).values
            
            p_observation_given_utterance = normalize(
                np.stack([
                    verify(
                        *a, 
                        possible_observations,
                        n_answers=possible_observations.max(),
                        n_students=possible_observations.shape[1]
                    )
                    for a in possible_utterances
                ]), 
                1
            )
            
            # dimensions (participant, utterance, observation)
            # which store the production probabilities 
            # in that size condition
            p_utterance_given_observation = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance,
                costs,
                alpha[i]
            )
            
            # shape: (datapoint in condition, possible utterance)
            production_probs_masked = p_utterance_given_observation[
                # index the right participant for each datapoint
                data.loc[submask, 'id'],
                # consider all utterances
                :,
                # set them to the corresponding 
                # production probabilities
                data.loc[submask,'index_observation']
            ]
                        
            p_production = tt.set_subtensor(
                p_production[
                    # only the datapoints 
                    # in the condition of interest
                    submask,
                    # all possible utterances
                    :
                ],
                production_probs_masked
            )
                    
        pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_base_hierarchical


########## lr argstrength

def factory_model_lr_argstrength(data, list_possible_observations, 
                                 possible_utterances, include_observed=True,
                                 include_S1=False):
                               
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)
    
    with pm.Model() as model_lr_argstrength:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=5,
            # one alpha per size array condition
            shape=len(list_possible_observations)
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
        
        
        # define array to accumulate production probabilities
        p_production = tt.ones((
            len(possible_utterances), 
            len(data)
        ))
        
        for i, possible_observations in enumerate(list_possible_observations):

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
                verify(
                    *a, 
                    possible_observations,
                    n_answers=possible_observations.max(),
                    n_students=possible_observations.shape[1]
                )
                for a in possible_utterances
            ])

            # literal listener
            p_observation_given_utterance = normalize(
                utterance_observation_compatibility,
                1
            )
            
            p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i],
                beta,
                argumentative_strengths_positive
            )

            p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i], 
                beta,
                argumentative_strengths_negative
            )

            p_utterance_given_observation = tt.stack((
                p_utterance_given_observation_low,
                p_utterance_given_observation_high
            ))
            
            submask = (data['index_array_size_condition']==i).values
            
            production_probs_masked = p_utterance_given_observation[
                data.loc[submask,'condition'],
                # consider all utterances
                :,
                # set them to the corresponding 
                # production probabilities
                data.loc[submask,'index_observation']
            ].T

            p_production = tt.set_subtensor(
                p_production[
                    # all utterances
                    :,
                    # only the datapoints 
                    # in the condition of interest
                    submask
                ],
                production_probs_masked
            )

            if include_S1:
                pm.Deterministic(
                    f'S1_{i}',
                    p_utterance_given_observation
                )

        utterances = pm.Categorical(
            'utterances',
            p_production.T,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_lr_argstrength

def factory_model_lr_argstrength_hierarchical(data, list_possible_observations, 
                                              possible_utterances, include_observed=True):
    
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)

    with pm.Model() as model_lr_argstrength_hierarchical:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=1,
            # one alpha per size array condition
            shape=(len(list_possible_observations),1)
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1,
            # one alpha per size array condition
            shape=(len(list_possible_observations),1)
        )
        
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(1,data.id.max()+1)
        )
        
        # shape (array size condition, participant)
        alpha = pm.Deterministic(
            f'alpha',
            pm.math.invlogit(
                alpha_mu + 
                alpha_sigma * alpha_zs
            ) * 5
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
        
        beta_offset = pm.Normal(
            'beta_offset',
            # number of participants
            shape=(data.id.max()+1,)
        )
        beta = pm.Deterministic(
            'beta',
            # move from unconstrained space to constrained space
            pm.math.invlogit(beta_mu + beta_offset * beta_sigma)
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )
        
        costs = mask_none * cost

        
        # define array to accumulate production probabilities
        # dimensions: (datapoint, utterance)
        p_production = tt.ones((
            len(data),
            len(possible_utterances)
        ))

        for i, possible_observations in enumerate(list_possible_observations):
            
            submask = (data['index_array_size_condition']==i).values

            p_observation_given_utterance = normalize(
                np.stack([
                    verify(
                        *a, 
                        possible_observations,
                        n_answers=possible_observations.max(),
                        n_students=possible_observations.shape[1]
                    )
                    for a in possible_utterances
                ]), 
                1
            )
            
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
            
            p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i],
                beta,
                argumentative_strengths_positive
            )

            p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i], 
                beta,
                argumentative_strengths_negative
            )
            
            # dimensions: (condition, participant, utterance, observation)
            p_utterance_given_observation = tt.stack((
                p_utterance_given_observation_low,
                p_utterance_given_observation_high
            ))
                        
            production_probs_masked = p_utterance_given_observation[
                # consider high and low conditions
                data.loc[submask,'condition'],
                # index the right participant for each datapoint
                data.loc[submask, 'id'],
                # consider all utterances
                :,
                # set them to the corresponding 
                # production probabilities
                data.loc[submask,'index_observation']
            ]
            
            p_production = tt.set_subtensor(
                p_production[
                    # only the datapoints 
                    # in the condition of interest
                    submask,
                    # all utterances
                    :
                ],
                production_probs_masked
            )

        utterances = pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_lr_argstrength_hierarchical


############## maximin argstrength

def factory_model_maximin_argstrength(data, list_possible_observations, 
                                      possible_utterances, include_observed=True,
                                      include_S1=False):
                                   
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)
    
    with pm.Model() as model_maximin_argstrength:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=5,
            # one alpha per size array condition
            shape=len(list_possible_observations)
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
        
        # define array to accumulate production probabilities
        p_production = tt.ones((
            len(possible_utterances), 
            len(data)
        ))
        
        for i, possible_observations in enumerate(list_possible_observations):

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
                verify(
                    *a, 
                    possible_observations,
                    n_answers=possible_observations.max(),
                    n_students=possible_observations.shape[1]
                )
                for a in possible_utterances
            ])

            # literal listener
            p_observation_given_utterance = normalize(
                utterance_observation_compatibility,
                1
            )
            
            p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i],
                beta,
                maximin_argstrengths_positive
            )

            p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i], 
                beta,
                maximin_argstrengths_negative
            )

            p_utterance_given_observation = tt.stack((
                p_utterance_given_observation_low,
                p_utterance_given_observation_high
            ))
            
            submask = (data['index_array_size_condition']==i).values
            
            production_probs_masked = p_utterance_given_observation[
                data.loc[submask,'condition'],
                # consider all utterances
                :,
                # set them to the corresponding 
                # production probabilities
                data.loc[submask,'index_observation']
            ].T

            p_production = tt.set_subtensor(
                p_production[
                    # all utterances
                    :,
                    # only the datapoints 
                    # in the condition of interest
                    submask
                ],
                production_probs_masked
            )

            if include_S1:
                pm.Deterministic(
                    f'S1_{i}',
                    p_utterance_given_observation
                )

        pm.Categorical(
            'utterances',
            p_production.T,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )

    return model_maximin_argstrength

def factory_model_maximin_argstrength_hierarchical(data, list_possible_observations, 
                                                   possible_utterances, include_observed=True):
    
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)

    with pm.Model() as model_maximin_argstrength_hierarchical:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=1,
            # one alpha per size array condition
            shape=(len(list_possible_observations),1)
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1,
            shape=(len(list_possible_observations),1)
        )
        
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(1,data.id.max()+1)
        )
        
        # shape (array size condition, participant)
        alpha = pm.Deterministic(
            'alpha',
            pm.math.invlogit(  
                alpha_mu + 
                alpha_sigma * alpha_zs
            ) * 2
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
        
        beta_offset = pm.Normal(
            'beta_offset',
            # number of participants
            shape=(data.id.max()+1,)
        )
        
        beta = pm.Deterministic(
            'beta',
            # move from unconstrained space to constrained space
            pm.math.invlogit(beta_mu + beta_offset * beta_sigma)
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )
        
        costs = mask_none * cost
        
        # define array to accumulate production probabilities
        # dimensions: (datapoint, utterance)
        p_production = tt.ones((
            len(data),
            len(possible_utterances)
        ))

        for i, possible_observations in enumerate(list_possible_observations):
            
            submask = (data['index_array_size_condition']==i).values

            p_observation_given_utterance = normalize(
                np.stack([
                    verify(
                        *a, 
                        possible_observations,
                        n_answers=possible_observations.max(),
                        n_students=possible_observations.shape[1]
                    )
                    for a in possible_utterances
                ]), 
                1
            )
            
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
            
            p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i],
                beta,
                argumentative_strengths_positive
            )

            p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i], 
                beta,
                argumentative_strengths_negative
            )
            
            # dimensions: (condition, participant, utterance, observation)
            p_utterance_given_observation = tt.stack((
                p_utterance_given_observation_low,
                p_utterance_given_observation_high
            ))
                        
            production_probs_masked = p_utterance_given_observation[
                # consider high and low conditions
                data.loc[submask,'condition'],
                # index the right participant for each datapoint
                data.loc[submask, 'id'],
                # consider all utterances
                :,
                # set them to the corresponding 
                # production probabilities
                data.loc[submask,'index_observation']
            ]
            
            p_production = tt.set_subtensor(
                p_production[
                    # only the datapoints 
                    # in the condition of interest
                    submask,
                    # all utterances
                    :
                ],
                production_probs_masked
            )

        utterances = pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_maximin_argstrength_hierarchical


######### pragmatic argstrength

def factory_model_prag_argstrength(data, list_possible_observations, 
                                   possible_utterances, include_observed=True,
                                   include_S1=False):

    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)

    with pm.Model() as model_prag_argstrength:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=5,
            # one alpha per size array condition
            shape=len(list_possible_observations)
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
        
        # define array to accumulate production probabilities
        p_production = tt.ones((
            len(possible_utterances), 
            len(data)
        ))
        
        for i, possible_observations in enumerate(list_possible_observations):
            
            prag_argstrengths_positive = theano_calculate_pragmatic_argstrength(
                possible_utterances, 
                possible_observations,
                gamma_prove=0.85, 
                gamma_disprove=0.15,
                alpha=alpha[i],
                costs=costs,
            )

            prag_argstrengths_negative = theano_calculate_pragmatic_argstrength(
                possible_utterances, 
                possible_observations,
                gamma_prove=0.15, 
                gamma_disprove=0.85,
                alpha=alpha[i],
                costs=costs,
            )
            
            utterance_observation_compatibility = np.stack([
                verify(
                    *a, 
                    possible_observations,
                    n_answers=possible_observations.max(),
                    n_students=possible_observations.shape[1]
                )
                for a in possible_utterances
            ])

            # literal listener
            p_observation_given_utterance = normalize(
                utterance_observation_compatibility,
                1
            )
            
            p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i], 
                beta,
                prag_argstrengths_positive
            )

            p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i], 
                beta,
                prag_argstrengths_negative
            )

            p_utterance_given_observation = tt.stack((
                p_utterance_given_observation_low,
                p_utterance_given_observation_high
            ))
            
            submask = (data['index_array_size_condition']==i).values
            
            production_probs_masked = p_utterance_given_observation[
                data.loc[submask,'condition'],
                # consider all utterances
                :,
                # set them to the corresponding 
                # production probabilities
                data.loc[submask,'index_observation']
            ].T

            p_production = tt.set_subtensor(
                p_production[
                    # all utterances
                    :,
                    # only the datapoints 
                    # in the condition of interest
                    submask
                ],
                production_probs_masked
            )

            if include_S1:
                pm.Deterministic(
                    f'S1_{i}',
                    p_utterance_given_observation
                )

        pm.Categorical(
            'utterances',
            p_production.T,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_prag_argstrength

def factory_model_prag_argstrength_hierarchical(data, list_possible_observations, 
                                   possible_utterances, include_observed=True):

    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)

    with pm.Model() as model_prag_argstrength_hierarchical:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=1,
            shape=(len(list_possible_observations),1)
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1,
            shape=(len(list_possible_observations),1)
        )
        
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(1,data.id.max()+1)
        )
        
        # shape (array size condition, participant)
        alpha = pm.Deterministic(
            f'alpha',
            pm.math.invlogit(
                alpha_mu + 
                alpha_sigma * alpha_zs
            ) * 5
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
        
        beta_offset = pm.Normal(
            'beta_offset',
            # number of participants
            shape=(data.id.max()+1,)
        )
        beta = pm.Deterministic(
            'beta',
            # move from unconstrained space to constrained space
            pm.math.invlogit(beta_mu + beta_offset * beta_sigma)
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )
        
        costs = mask_none * cost

        
        # define array to accumulate production probabilities
        # dimensions: (datapoint, utterance)
        p_production = tt.ones((
            len(data),
            len(possible_utterances)
        ))

        for i, possible_observations in enumerate(list_possible_observations):
            
            submask = (data['index_array_size_condition']==i).values

            p_observation_given_utterance = normalize(
                np.stack([
                    verify(
                        *a, 
                        possible_observations,
                        n_answers=possible_observations.max(),
                        n_students=possible_observations.shape[1]
                    )
                    for a in possible_utterances
                ]), 
                1
            )
            
            argumentative_strengths_positive = theano_calculate_pragmatic_argstrength(
                possible_utterances, 
                possible_observations,
                gamma_prove=0.85, 
                gamma_disprove=0.15,
                alpha=alpha[i],
                costs=costs
            )

            argumentative_strengths_negative = theano_calculate_pragmatic_argstrength(
                possible_utterances, 
                possible_observations,
                gamma_prove=0.15, 
                gamma_disprove=0.85,
                alpha=alpha[i],
                costs=costs
            )
            
            p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i],
                beta,
                argumentative_strengths_positive
            )

            p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i], 
                beta,
                argumentative_strengths_negative
            )
            
            # dimensions: (condition, participant, utterance, observation)
            p_utterance_given_observation = tt.stack((
                p_utterance_given_observation_low,
                p_utterance_given_observation_high
            ))
                        
            production_probs_masked = p_utterance_given_observation[
                # consider high and low conditions
                data.loc[submask,'condition'],
                # index the right participant for each datapoint
                data.loc[submask, 'id'],
                # consider all utterances
                :,
                # set them to the corresponding 
                # production probabilities
                data.loc[submask,'index_observation']
            ]
            
            p_production = tt.set_subtensor(
                p_production[
                    # only the datapoints 
                    # in the condition of interest
                    submask,
                    # all utterances
                    :
                ],
                production_probs_masked
            )

        utterances = pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_prag_argstrength_hierarchical



########### nonparametric argstrength


def factory_model_nonparametric_argstrength(data, list_possible_observations, 
                                            possible_utterances, include_observed=True,
                                            include_S1=False):
                               
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)
    
    with pm.Model() as model_nonparametric_argstrength:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=5,
            # one alpha per size array condition
            shape=len(list_possible_observations)
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
        
        
        # define array to accumulate production probabilities
        p_production = tt.ones((
            len(possible_utterances), 
            len(data)
        ))
        
        for i, possible_observations in enumerate(list_possible_observations):

            argumentative_strengths_positive = calculate_nonparametric_argstrength(
                possible_utterances, 
                possible_observations, 
                condition='high'
            ).flatten()
            
            argumentative_strengths_negative = calculate_nonparametric_argstrength(
                possible_utterances, 
                possible_observations, 
                condition='low'
            ).flatten()

            utterance_observation_compatibility = np.stack([
                verify(
                    *a, 
                    possible_observations,
                    n_answers=possible_observations.max(),
                    n_students=possible_observations.shape[1]
                )
                for a in possible_utterances
            ])

            # literal listener
            p_observation_given_utterance = normalize(
                utterance_observation_compatibility,
                1
            )
            
            p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i],
                beta,
                argumentative_strengths_positive
            )

            p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i], 
                beta,
                argumentative_strengths_negative
            )

            p_utterance_given_observation = tt.stack((
                p_utterance_given_observation_low,
                p_utterance_given_observation_high
            ))
            
            submask = (data['index_array_size_condition']==i).values
            
            production_probs_masked = p_utterance_given_observation[
                data.loc[submask,'condition'],
                # consider all utterances
                :,
                # set them to the corresponding 
                # production probabilities
                data.loc[submask,'index_observation']
            ].T

            p_production = tt.set_subtensor(
                p_production[
                    # all utterances
                    :,
                    # only the datapoints 
                    # in the condition of interest
                    submask
                ],
                production_probs_masked
            )

            if include_S1:
                pm.Deterministic(
                    f'S1_{i}',
                    p_utterance_given_observation
                )

        utterances = pm.Categorical(
            'utterances',
            p_production.T,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_nonparametric_argstrength


def factory_model_nonparametric_argstrength_hierarchical(data, list_possible_observations, 
                                                         possible_utterances, include_observed=True):
    
    mask_none = np.any(possible_utterances=='none', axis=1).astype(int)

    with pm.Model() as model_nonparametric_argstrength_hierarchical:

        data_utterance = pm.Data(
            'observed', 
            data['index_utterance']
        )

        alpha_mu = pm.Normal(
            'alpha_mu', 
            mu=0,
            sigma=1,
            shape=(len(list_possible_observations),1)
        )
        
        alpha_sigma = pm.HalfNormal(
            'alpha_sigma',
            sigma=1,
            shape=(len(list_possible_observations),1)
        )
        
        alpha_zs = pm.Normal(
            'alpha_zs',
            shape=(1,data.id.max()+1)
        )
        
        # shape (array size condition, participant)
        alpha = pm.Deterministic(
            f'alpha',
            pm.math.invlogit(
                alpha_mu + 
                alpha_sigma * alpha_zs
            ) * 5
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
        
        beta_offset = pm.Normal(
            'beta_offset',
            # number of participants
            shape=(data.id.max()+1,)
        )
        beta = pm.Deterministic(
            'beta',
            # move from unconstrained space to constrained space
            pm.math.invlogit(beta_mu + beta_offset * beta_sigma)
        )

        cost = pm.Exponential(
            'costnone',
            lam=0.5
        )
        
        costs = mask_none * cost

        
        # define array to accumulate production probabilities
        # dimensions: (datapoint, utterance)
        p_production = tt.ones((
            len(data),
            len(possible_utterances)
        ))

        for i, possible_observations in enumerate(list_possible_observations):
            
            submask = (data['index_array_size_condition']==i).values

            p_observation_given_utterance = normalize(
                np.stack([
                    verify(
                        *a, 
                        possible_observations,
                        n_answers=possible_observations.max(),
                        n_students=possible_observations.shape[1]
                    )
                    for a in possible_utterances
                ]), 
                1
            )
            
            argumentative_strengths_positive = calculate_nonparametric_argstrength(
                possible_utterances, 
                possible_observations, 
                condition='high'
            ).flatten()
            
            argumentative_strengths_negative = calculate_nonparametric_argstrength(
                possible_utterances, 
                possible_observations, 
                condition='low'
            ).flatten()
            
            p_utterance_given_observation_high = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i],
                beta,
                argumentative_strengths_positive
            )

            p_utterance_given_observation_low = theano_calculate_pragmatic_speaker(
                p_observation_given_utterance, 
                costs,
                alpha[i], 
                beta,
                argumentative_strengths_negative
            )
            
            # dimensions: (condition, participant, utterance, observation)
            p_utterance_given_observation = tt.stack((
                p_utterance_given_observation_low,
                p_utterance_given_observation_high
            ))
                        
            production_probs_masked = p_utterance_given_observation[
                # consider high and low conditions
                data.loc[submask,'condition'],
                # index the right participant for each datapoint
                data.loc[submask, 'id'],
                # consider all utterances
                :,
                # set them to the corresponding 
                # production probabilities
                data.loc[submask,'index_observation']
            ]
            
            p_production = tt.set_subtensor(
                p_production[
                    # only the datapoints 
                    # in the condition of interest
                    submask,
                    # all utterances
                    :
                ],
                production_probs_masked
            )

        utterances = pm.Categorical(
            'utterances',
            p_production,
            observed=data_utterance if include_observed else None,
            shape=len(data['index_utterance'])
        )
    
    return model_nonparametric_argstrength_hierarchical