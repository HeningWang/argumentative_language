from functions.helper_functions import verify, normalize

from functions.argstrengths import (
    calculate_argumentative_strength,
    calculate_maximin_argstrength,
)

from functions.argstrengths_fullstatespace import (
    calculate_argumentative_strength_fullstatespace,
    calculate_maximin_argstrength_fullstatespace,
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns


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

def plot_compare_argstrength_maximin(gamma_for, 
                                     gamma_against,
                                     possible_utterances, 
                                     possible_observations, 
                                     fullstatespace=False):
    
    if fullstatespace:
        maximin_argstrength_high = calculate_maximin_argstrength_fullstatespace(
            possible_utterances, possible_observations, 
            gamma_for, gamma_against
        )
        maximin_argstrength_low = calculate_maximin_argstrength_fullstatespace(
            possible_utterances, possible_observations, 
            gamma_against, gamma_for
        )
        argumentative_strengths_positive = calculate_argumentative_strength_fullstatespace(
            possible_utterances, possible_observations, 
            gamma_for, gamma_against
        )
        argumentative_strengths_negative = calculate_argumentative_strength_fullstatespace(
            possible_utterances, possible_observations, 
            gamma_against, gamma_for
        )

    else:
        # high condition
        maximin_argstrength_high = calculate_maximin_argstrength(
            possible_utterances, possible_observations, 
            gamma_for, gamma_against
        )
        # low condition
        maximin_argstrength_low = calculate_maximin_argstrength(
            possible_utterances, possible_observations, 
            gamma_against, gamma_for
        )
        argumentative_strengths_positive = calculate_argumentative_strength(
            possible_utterances, possible_observations,
            gamma_for, gamma_against
        )
        argumentative_strengths_negative = calculate_argumentative_strength(
            possible_utterances, possible_observations,
        gamma_against, gamma_for
        )
    
    df = pd.DataFrame(
        possible_utterances,
        columns=['q1', 'q2', 'adj']
    )

    df['minimax_high'] = maximin_argstrength_high
    df['minimax_low'] = maximin_argstrength_low
    df['minimax_delta'] = maximin_argstrength_high - maximin_argstrength_low

    df['lr_high'] = argumentative_strengths_positive
    df['lr_low'] = argumentative_strengths_negative

    df = df.sort_values('minimax_delta').reset_index(drop=True)
    df = df.drop(columns='minimax_delta',)

    # ax = df.plot(
    #     kind='barh', 
    #     # subplots=True, 
    #     yticks=np.arange(len(df)),
    #     # legend=False,
    #     figsize=(6,15),
    #     layout=(1,5),
    #     sharey=True
    # )
    
    ax = df.plot(
        kind='bar', 
        # subplots=True, 
        xticks=np.arange(len(df)),
        # legend=False,
        figsize=(15,6),
        layout=(5,1),
        sharex=True
    )

    # for ax in axes.flatten():
    # ax.set_yticklabels(df.iloc[:,:3].apply(lambda x: '|'.join(x), 1))
    ax.set_xticklabels(df.iloc[:,:3].apply(lambda x: '|'.join(x), 1))
    plt.xticks(rotation=75)

    plt.tight_layout()
    plt.show()


def scatter_hist(x, y, ax, ax_histx, ax_histy, color, label, kwargs_scatter=dict(), kwargs_hist=dict()):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, color=color, label=label, **kwargs_scatter)
    ax_histx.hist(x, bins=30, density=True, color=color, **kwargs_hist)
    ax_histy.hist(y, bins=30, orientation='horizontal', density=True, color=color, **kwargs_hist)


def plot_marginal_deltas(data, info_deltas, arg_deltas_high, arg_deltas_low):
    
    info_deltas_pointwise = info_deltas[
        data['index_utterance'], 
        data['index_observation']
    ]

    arg_deltas_pointwise = np.where(
        data['condition'],
        arg_deltas_high[data['index_utterance'], data['index_observation']],
        arg_deltas_low[data['index_utterance'], data['index_observation']]
    )

    possible_argdeltas = np.where(
        data['condition'],
        arg_deltas_high[:, data['index_observation']],
        arg_deltas_low[:, data['index_observation']]
    ).flatten()

    possible_infodeltas = info_deltas[:, data['index_observation']].flatten()

    possible_values = np.column_stack((possible_infodeltas, possible_argdeltas))
    possible_values = possible_values[np.all(possible_values!=np.inf,1)]

    # start with a square Figure
    fig = plt.figure(figsize=(7, 3))

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    rect_scatter = [left, bottom, width, height]
    ax = fig.add_axes(rect_scatter)

    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # definitions for the axes
    spacing = 0.22
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ax_histx_1 = fig.add_axes(rect_histx, sharex=ax)
    ax_histy_1 = fig.add_axes(rect_histy, sharey=ax)

    scatter_hist(
        x=arg_deltas_pointwise,
        y=info_deltas_pointwise,
        ax=ax, 
        ax_histx=ax_histx_1, 
        ax_histy=ax_histy_1,
        color='red',
        label='data',
        kwargs_scatter={
            'alpha': 0.1,
            's': 12
        }
    )

    scatter_hist(
        x=possible_values.T[1],
        y=possible_values.T[0],
        ax=ax, 
        ax_histx=ax_histx, 
        ax_histy=ax_histy,
        color='blue',
        label='possible',
        kwargs_scatter={
            'facecolor': 'none',
            'edgecolor': 'blue',
            's': 13
        }
    )

    ax.legend(
        bbox_to_anchor=(1.1, 1.1),
        bbox_transform=fig.transFigure
    )

    ax.set_xlabel(r'Arg $\Delta$')
    ax.set_ylabel(r'Info $\Delta$')

    sns.despine(ax=ax_histx, left=True)
    sns.despine(ax=ax_histx_1, left=True)
    ax_histx.set_yticks([])
    ax_histx_1.set_yticks([])

    sns.despine(ax=ax_histy, bottom=True)
    sns.despine(ax=ax_histy_1, bottom=True)
    ax_histy.set_xticks([])
    ax_histy_1.set_xticks([])

    ax_histy.set_xlabel('possible')
    ax_histy_1.set_xlabel('data')

    ax_histx.set_ylabel('possible', labelpad=20, rotation=45)
    ax_histx_1.set_ylabel('data', rotation=45)
                         
    return fig, ax, ax_histx, ax_histy


def plot_obswise_deltas(data, info_deltas, arg_deltas_high, arg_deltas_low, fname=None):
    
    info_deltas_pointwise = info_deltas[data['index_utterance'], data['index_observation']]

    arg_deltas_pointwise = np.where(
        data['condition'],
        arg_deltas_high[data['index_utterance'], data['index_observation']],
        arg_deltas_low[data['index_utterance'], data['index_observation']]
    )

    obs = [
        '|'.join(np.array(x).astype(str))
        for x 
        in data['row_number']
    ]

    df_plot = pd.DataFrame({
        'argdeltas': arg_deltas_pointwise,
        'infodeltas': info_deltas_pointwise,
        'obs': obs
    })

    # with sns.plotting_context("paper"):
    g = sns.FacetGrid(
        data=df_plot,
        col='obs',
        col_wrap=7,
        height=1.5,
        col_order=sorted(np.unique(obs), key=lambda x: np.array(x.split('|')).astype(int).sum())
    )

    g.map_dataframe(
        sns.scatterplot, 
        x="infodeltas", 
        y="argdeltas",
        alpha=0.2
    )

    g.set_titles(
        # row_template = '{row_name}', 
        col_template = '{col_name}'
    )

    g.set_xlabels(r'Info$\Delta$')
    g.set_ylabels(r'Arg$\Delta$')
    
    if fname is not None:
        plt.savefig(fname, dpi=300)
    else:
        plt.show()


def plot_MAP_recovery(prior_pd, MAP_values, figsize=(7,3)):
    
    # get non transformed variable names
    variables = [
        k 
        for k in prior_pd.keys() 
        if not pm.util.is_transformed_name(k)
        if k != 'utterances'
    ]
    
    fig, axes = plt.subplots(1,len(variables), figsize=figsize)
    
    for ax, variable in zip(axes, variables):
        ax.scatter(
            prior_pd[variable], 
            [a[variable] for a in MAP_values]
        )
        ax.set_title(variable)

    axes[0].set_ylabel('MAP')
    # draw y=x line for each ax
    for ax in axes:
        ax.set_xlabel('real value')
        xpoints = ypoints = ax.get_xlim()
        ax.plot(
            xpoints, 
            ypoints, 
            linestyle='--', 
            color='k', 
            lw=1, 
            scalex=False, 
            scaley=False
        )

    plt.tight_layout()
    plt.show()
