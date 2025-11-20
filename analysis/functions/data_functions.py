import pandas as pd
import numpy as np
from itertools import product
from functions.helper_functions import verify


def get_and_clean_data_exp1(pathdata='data_raw.csv'):
    """
    This functions works for the pilot and exp1
    
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

    raw_data = pd.read_csv(pathdata)

    try:
        # this is the keys for the experimental data
        data = raw_data[[
            'condition', 
            'responses',       # previous: response
            'studentsArray',   # previous: row_number
            # 'trial_name',      # doesn't exist for this version
            'submission_id'     # previous: prolific_id
        ]]

        # change to the names I've used so far
        data.rename(columns={
            'responses': 'response',
            'studentsArray': 'row_number',
            'submission_id': 'prolific_id'
        }, inplace=True)

        data.loc[:,'response'] = data.response.str.lower()

        data_observations = raw_data['studentsArray']

        data.loc[:,'condition'] = (data.condition == 1).astype(int)


    except KeyError:

        # this is the keys for the pilot data
        # (older version of magpie)
        data = raw_data[[
            'condition', 
            'response', 
            'row_number', 
            'trial_name', 
            'prolific_id'
        ]]

        data = (
            data[raw_data['trial_name']=='main_trials']
            .reset_index(drop=True)
        )

        data_observations = raw_data['row_number']

        print(
            'a total of ' +
            str(len(data[raw_data['trial_name'] != 'main_trials'])) +
            ' of the raw datapoints are test trials. ' +
            'This leaves 20 potential datapoints per participant.'
        )

        data.loc[:,'condition'] = (data.condition == 'high').astype(int)

    data.loc[:,'row_number'] = (
        data
        .row_number
        .str
        .split('|')
        .apply((
            lambda x: [int(a) for a in x]
        ))
    )

    data.loc[:,'response'] = (
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
            for a in data_observations.unique()
        ], 
        dtype=int
    )

    possible_utterances = np.array(list(product(qs, qs, adjs)))

    index_observations_data = data.row_number.apply(
        lambda observation: np.argwhere(
            (possible_observations==observation).all(1)
        ).flatten()[0]
    )
    data.loc[:,'index_observation'] = index_observations_data

    index_utterance_data = data.response.apply(
        lambda utterance: np.argwhere(
            (possible_utterances==utterance).all(1)
        ).flatten()[0]
    )
    data.loc[:,'index_utterance'] = index_utterance_data

    # exclude the false responses from the data
    utterance_observation_compatibility = np.stack([
        verify(
            *a, 
            possible_observations, 
            n_answers=possible_observations.max(), 
            n_students=possible_observations.shape[1]
        )
        for a in possible_utterances
    ])

    data['false'] = 1 - utterance_observation_compatibility[
        data.index_utterance, 
        data.index_observation
    ].astype(bool)


    participants_to_exclude = (
        data
        .groupby('prolific_id')
        .sum('false')
        ['false']
        > 4
    )

    print(
        participants_to_exclude.sum(),
        ' of the participants were excluded as'
        ' they gave more than 4 false responses'
    )

    # go from values grouped by participant
    # to ids in the full dataset
    ids_to_exclude = participants_to_exclude[participants_to_exclude].index.values
    index_to_exclude_data = (
        data['prolific_id']
        .values
        .reshape(-1,1) 
        == ids_to_exclude
    ).any(1)

    data = data[~index_to_exclude_data]

    print(
        data['false'].sum(),
        ' of the observations in the included participants'
        ' were excluded because literally false'
    )

    # only keep true observations
    data = data[~data['false'].astype(bool)]

    # record the index of the participant
    _, participant_id = np.unique(
        data['prolific_id'], 
        return_inverse=True
    )
    data.loc[:,'id'] = participant_id
    data = data.drop('prolific_id', axis=1)
    data = data.drop('false', axis=1)
    
    return raw_data, data, possible_observations, possible_utterances


def get_and_clean_data_exp2(pathdata='data_raw.csv', pathdata_firstexp=None, select_condition=None):
    """
    Get the raw data and cleans it. 
    Returns various useful objects (see below).
    
    NOTE: This is rewritten to work with the second experiment
    
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

    ########### Get data, rename columns, create columns
    
    raw_data = pd.read_csv(pathdata)
    
    # add the data from the first experiment
    if pathdata_firstexp is not None:
        raw_data_firstexp = pd.read_csv(pathdata_firstexp)
        raw_data_firstexp.loc[:,'arraySizeCondition'] = 'wideShort'

        raw_data = pd.concat((
            raw_data_firstexp.reset_index(drop=True), 
            raw_data.reset_index(drop=True)
        ))
        
    if select_condition is not None:
        raw_data = raw_data[
            raw_data['arraySizeCondition'] == select_condition
        ] 
    
    raw_data = raw_data[raw_data['prolific_study_id'] != 'UNSAVED-STUDY']
    
    # this is the keys for the experimental data
    data = raw_data[[
        'condition', 
        'responses',       # previous: response
        'studentsArray',   # previous: row_number
        'submission_id',     # previous: prolific_id
        'arraySizeCondition'
    ]].reset_index(drop=True)

    # change to the names I've used in the previous notebook
    # for consistence
    data.rename(columns={
        'responses': 'response',
        'studentsArray': 'row_number',
        'submission_id': 'prolific_id',
        'arraySizeCondition': 'array_size_condition'
    }, inplace=True)

    data.loc[:,'response'] = data.response.str.lower()
    data_observations = raw_data['studentsArray']
    data.loc[:,'condition'] = (data.condition == 1).astype(int)
    data.loc[:,'index_array_size_condition'] = data['array_size_condition'].factorize()[0]

    ######## Factorize size array conditions and get arrays
    
    grouped = data.groupby('index_array_size_condition')['row_number']

    # get the unique values and their observed indices in data,
    # but grouped by sizeArrayCondition
    data.loc[:,'index_observation'] = (
        grouped
        .transform(lambda x: pd.factorize(x)[0])
    )

    # get for each observation the factorized index of 
    # the observation _for that condition_
    possible_observations_by_condition = (
        grouped
        .apply(lambda x: pd.factorize(x)[1].values)
    )
    
    list_possible_observations = (
        possible_observations_by_condition
        .apply(
            lambda x: np.array([
                a.split('|') for a in x
            ]).astype(int)
        )
        .to_list()
    )
    
    data.loc[:,'row_number'] = (
        data
        .row_number
        .str
        .split('|')
        .apply((
            lambda x: [int(a) for a in x]
        ))
    )

    data.loc[:,'response'] = (
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

    possible_utterances = np.array(list(product(qs, qs, adjs)))

    index_utterance_data = data.response.apply(
        lambda utterance: np.argwhere(
            (possible_utterances==utterance).all(1)
        ).flatten()[0]
    )
    data.loc[:,'index_utterance'] = index_utterance_data
    
    # exclude the false responses from the data
    list_utterance_observation_compatibility = np.stack([
        np.stack([
            verify(
                *a, 
                possible_observations,
                n_answers=possible_observations.max(),
                n_students=possible_observations.shape[1]
            )
            for a in possible_utterances
        ])
        for possible_observations in list_possible_observations
    ])

    data.loc[:,'false'] = 1 - list_utterance_observation_compatibility[
        data.index_array_size_condition,
        data.index_utterance, 
        data.index_observation
    ].astype(bool)

    participants_to_exclude = (
        data
        .groupby('prolific_id')
        .sum('false')
        ['false']
        > 4
    )

    print(
        participants_to_exclude.sum(),
        ' of the participants were excluded as'
        ' they gave more than 4 false responses'
    )

    # go from values grouped by participant
    # to ids in the full dataset
    ids_to_exclude = participants_to_exclude[participants_to_exclude].index.values
    index_to_exclude_data = (
        data['prolific_id']
        .values
        .reshape(-1,1) 
        == ids_to_exclude
    ).any(1)

    data = data[~index_to_exclude_data]

    print(
        data['false'].sum(),
        ' of the observations in the included participants'
        ' were excluded because literally false'
    )

    # only keep true observations
    data = data[~data['false'].astype(bool)]

    # record the index of the participant
    _, participant_id = np.unique(
        data['prolific_id'], 
        return_inverse=True
    )
    data.loc[:,'id'] = participant_id
    data = data.drop('prolific_id', axis=1)
    data = data.drop('false', axis=1)
    
    return raw_data, data.reset_index(drop=True), list_possible_observations, possible_utterances

