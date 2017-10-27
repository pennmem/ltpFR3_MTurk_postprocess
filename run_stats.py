import numpy as np
from pybeh.spc import spc
from pybeh.pnr import pnr
from pybeh.crp import crp
from pybeh.irt import irt
from scipy.stats import sem


def run_stats(d):
    """
    Use the data extracted from a psiTurk database to generate behavioral stats for each participant. Creates a stat
    data structure with the following format:
    
    stats = {
    subj1: { stat1: {condition1: value, condition2: value, ...}, 
            stat2: {condition1: value, condition2: value, ...}, 
            stat3: {condition1: value, condition2: value, ...}, 
            ... },
    subj2: { stat1: {condition1: value, condition2: value, ...}, 
            stat2: {condition1: value, condition2: value, ...}, 
            stat3: {condition1: value, condition2: value, ...}, 
            ... },
    subj3: { stat1: {condition1: value, condition2: value, ...}, 
            stat2: {condition1: value, condition2: value, ...}, 
            stat3: {condition1: value, condition2: value, ...}, 
            ... }, 
    ...
    }
    
    :param d: A data structure created by psiturk_tools.process_psiturk_data()
    :return: A dictionary containing a subdictionary for each participant. Each subdictionary contains all behavioral stats for that single participant.
    """
    stats = dict()

    for subj in d:
        list_iterator = range(len(d[subj]['serialpos']))

        # Extract subject, condition, recall, etc info from raw data to create recalls matrices, etc
        sub = np.array([subj for i in list_iterator])
        condi = [(d[subj]['list_len'][i], d[subj]['pres_rate'][i], str(d[subj]['pres_mod'][i]), d[subj]['dist_dur'][i]) for i in list_iterator]
        recalls = pad_into_array(d[subj]['serialpos']).astype(int)
        wasrec = np.array(d[subj]['recalled'])
        rt = pad_into_array(d[subj]['rt'])
        recw = pad_into_array(d[subj]['rec_words'])
        presw = pad_into_array(d[subj]['pres_words'])
        intru = recalls_to_intrusions(recalls)
        math = pad_into_array(d[subj]['math_correct']).astype(bool)

        # Run all stats for a single participant and add the resulting stats object to the stats dictionary
        stats[str(subj)] = stats_for_subj(sub, condi, recalls, wasrec, rt, recw, presw, intru, math)

    stats['average'] = avg_stats(stats)

    return stats


def stats_for_subj(sub, condi, recalls, wasrec, rt, recw, presw, intru, math):
    """
    Create a stats dictionary for a single participant.
    
    :param sub: A subject array for the stats calculations. As we are only running stats on one participant at a time, 
    this array will simply be [subj_name] * number_of_trials.
    :param condi: An array of tuples indicating which conditions were used on each trial (ll, pr, mod, dd)
    :param recalls: A matrix where item (i, j) is the serial position of the ith recall in trial j
    :param wasrec: A matrix where item (i, j) is 0 if the ith presented word in trial j was not recalled, 1 if it was
    :param rt: A matrix where item (i, j) is the response time (in ms) of the ith recall in trial j
    :param recw: A matrix where item (i, j) is the ith word recalled on trial j
    :param presw: A matrix where item (i, j) is the ith word presented on trial j
    :param intru: A list x items intrusions matrix (see recalls_to_intrusions)
    :return: 
    """
    stats_to_run = ['prec', 'spc', 'pfr', 'psr', 'ptr', 'crp_early', 'crp_late', 'plis', 'elis', 'reps', 'pli_recency']
    stats = {stat: {} for stat in stats_to_run}

    # filters = {'000': {'ll': 12, 'pr': 800, 'mod': 'a'}, '001': {'ll': 12, 'pr': 800, 'mod': 'v'},
    #           '010': {'ll': 12, 'pr': 1600, 'mod': 'a'}, '011': {'ll': 12, 'pr': 1600, 'mod': 'v'},
    #           '100': {'ll': 24, 'pr': 800, 'mod': 'a'}, '101': {'ll': 24, 'pr': 800, 'mod': 'v'},
    #           '110': {'ll': 24, 'pr': 1600, 'mod': 'a'}, '111': {'ll': 24, 'pr': 1600, 'mod': 'v'}}
    filters = {'12': {'ll': 12}, '24': {'ll': 24}, 'a12': {'ll': 12, 'mod': 'a'}, 'a24': {'ll': 24, 'mod': 'a'},
               'v12': {'ll': 12, 'mod': 'v'}, 'v24': {'ll': 24, 'mod': 'v'}, 'f12': {'ll': 12, 'pr': 800},
               'f24': {'ll': 24, 'pr': 800}, 's12': {'ll': 12, 'pr': 1600}, 's24': {'ll': 24, 'pr': 1600}}

    for f in filters:
        ll = filters[f]['ll']
        fsub, frecalls, fwasrec, frt, frecw, fpresw, fintru = [filter_by_condi(a, condi, **filters[f]) for a in [sub, recalls, wasrec, rt, recw, presw, intru]]

        # Calculate stats on all lists within the current condition
        stats['prec'][f] = prec(fwasrec[:, :ll], fsub)[0]
        stats['spc'][f] = spc(frecalls, fsub, ll)[0]
        stats['pfr'][f] = pnr(frecalls, fsub, ll, n=0)[0]
        stats['psr'][f] = pnr(frecalls, fsub, ll, n=1)[0]
        stats['ptr'][f] = pnr(frecalls, fsub, ll, n=2)[0]
        stats['crp_early'][f] = crp(frecalls[:, :3], fsub, ll, lag_num=3)[0]
        stats['crp_late'][f] = crp(frecalls[:, 2:], fsub, ll, lag_num=3)[0]
        #stats['irt'][f] = irt(frt)
        stats['plis'][f] = avg_pli(fintru, fsub, frecw)[0]
        stats['elis'][f] = avg_eli(fintru, fsub)[0]
        stats['reps'][f] = avg_reps(frecalls, fsub)[0]
        stats['pli_recency'][f] = pli_recency(fintru, fsub, 6, frecw)[0]

        # Fix CRPs to have a 0-lag of NaN
        stats['crp_early'][f][3] = np.nan
        stats['crp_late'][f][3] = np.nan

    stats['rec_per_trial'] = np.nanmean(wasrec, axis=1)
    stats['math_per_trial'] = np.sum(math, axis=1)

    return stats


def avg_stats(s):
    stats_to_run = ['prec', 'spc', 'pfr', 'psr', 'ptr', 'crp_early', 'crp_late',
                    'pli_early', 'pli_late', 'eli_early', 'eli_late', 'reps', 'pli_recency']
    filters = ['12', '24', 'a12', 'a24', 'v12', 'v24', 'f12', 'f24', 's12', 's24']
    excluded = ['MTK0019']

    avs = {}
    for stat in stats_to_run:
        avs[stat] = {}
        for f in filters:
            scores = []
            for subj in s:
                if s not in excluded:
                    scores.append(s[subj][stat][f])
            scores = np.array(scores)
            avs[stat][f] = np.nanmean(scores, axis=0)

    return avs


def filter_by_condi(a, condi, ll=None, pr=None, mod=None, dd=None):
    """
    Filter a matrix of trial data to only get the data from trials that match the specified condition(s)
    :param a: A numpy array with the data from one trial on each row
    :param condi: A list of tuples indicating the list length, presentation rate, modality, distractor duration of each trial
    :param ll: Return only trials with this list length condition (ignore if None)
    :param pr: Return only trials with this presentation rate condition (ignore if None)
    :param mod: Return only trials with this presentation modality condition (ignore if None)
    :param dd: Return only trials with this distractor duration condition (ignore if None
    :return: A numpy array containing only the data from trials that match the specified condition(s)
    """
    ind = [i for i in range(len(condi)) if ((ll is None or condi[i][0] == ll) and (pr is None or condi[i][1] == pr) and (mod is None or condi[i][2] == mod) and (dd is None or condi[i][3] == dd))]
    return a[ind]


def pad_into_array(l):
    """
    Turn an array of uneven lists into a numpy matrix by padding shorter lists with zeros. Modified version of a
    function by user Divakar on Stack Overflow, here:
    http://stackoverflow.com/questions/32037893/numpy-fix-array-with-rows-of-different-lengths-by-filling-the-empty-elements-wi

    :param l: A list of lists
    :return: A numpy array made from l, where all rows have been made the same length via padding
    """
    l = np.array(l)
    # Get lengths of each row of data
    lens = np.array([len(i) for i in l])

    # If l was empty, we can simply return the empty numpy array we just created
    if len(lens) == 0:
        return lens

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=l.dtype)
    out[mask] = np.concatenate(l)

    return out


def recalls_to_intrusions(rec):
    """
    Convert a recalls matrix to an intrusions matrix. In the recalls matrix, ELIs should be denoted by -999 and PLIs
    should be denoted by -n, where n is the number of lists back the word was originally presented. All positive numbers
    are assumed to be correct recalls. The resulting intrusions matrix denotes correct recalls by 0, ELIs by -1, and 
    PLIs by n, where n is the number of lists back the word was originally presented.
    
    :param rec: A lists x items recalls matrix, which is assumed to be a numpy array
    :return: A lists x items intrusions matrix
    """
    intru = rec.copy()
    # Set correct recalls to 0
    intru[np.where(intru > 0)] = 0
    # Convert negative numbers for PLIs to positive numbers
    intru *= -1
    # Convert ELIs to -1
    intru[np.where(intru == 999)] = -1
    return intru


def prec(recalled, subjects):
    """
    Calculate the overall probability of recall for each subject, given a lists x items matrix where 0s indicate
    words that were not subsequently recalled and 1s indicate words that were subsequently recalled. Item (i, j)
    should indicate whether the jth word presented in list i was recalled.
    
    :param recalled: A lists x items matrix, indicating whether each presented word was subsequently recalled
    :param subjects: A list of subject codes, indicating which subject produced each row of was_recalled
    :return: An array containing the overall probability of recall for each unique participant
    """
    if len(recalled) == 0:
        return np.array([]), np.array([])
    usub = np.unique(subjects)
    result = np.array([recalled[subjects == subj].mean() for subj in usub])
    stderr = np.array([sem(recalled[subjects == subj], nan_policy='omit') for subj in usub])

    return result, stderr


def pli_recency(intrusions, subjects, nmax, rec_words):
    """
    Calculate the ratio of PLIs that originated from 1 list back, 2 lists back, etc. up until nmax lists back.
    
    :param intrusions: An intrusions matrix in the format generated by recalls_to_intrusions
    :param subjects: A list of subject codes, indicating which subject produced each row of the intrusions matrix
    :param nmax: The maximum number of lists back to consider
    :param rec_words: A lists x words matrix, where item (i, j) is the jth word presented on the ith list
    :return: An array of length nmax, where item i is the ratio of PLIs that originated from i+1 lists back
    """
    if len(intrusions) == 0 or nmax < 1:
        return np.array([])

    usub = np.unique(subjects)
    result = np.zeros((len(usub), nmax+1), dtype=float)
    ll = len(intrusions[0])
    for i, s in enumerate(usub):
        for j in range(len(intrusions)):
            if subjects[j] == s:
                encountered = []
                for k in range(ll):
                    if intrusions[j][k] > 0 and rec_words[j][k] not in encountered:
                        encountered.append(rec_words[j][k])
                        if intrusions[j][k] <= nmax:
                            result[i, intrusions[j][k] - 1] += 1
                        else:
                            result[i, nmax] += 1

    result /= result.sum(axis=1)
    return result[:, :nmax]


def avg_pli(intrusions, subjects, rec_itemnos):
    """
    A modification of the behavioral toolbox's pli function. Calculate's each partcipant's average number of PLIs per 
    list instead of their total number of PLIs.
    
    :param intrusions: An intrusions matrix in the format generated by recalls_to_intrusions
    :param subjects: A list of subject codes, indicating which subject produced each row of the intrusions matrix
    :param rec_itemnos: A matrix in which each row is the list of IDs for all words recalled by a single subject on a
                        single trial. Rows are expected to be padded with 0s to all be the same length.
    :return: An array where each entry is the average number of PLIs per list for a single participant.
    """
    usub = np.unique(subjects)
    result = np.zeros(len(usub))
    for subject_index in range(len(usub)):
        count = 0.
        lists = 0.
        for subj in range(len(subjects)):
            if subjects[subj] == usub[subject_index]:
                lists += 1
                encountered = []
                for serial_pos in range(len(intrusions[0])):
                    if intrusions[subj][serial_pos] > 0 and rec_itemnos[subj][serial_pos] not in encountered:
                        count += 1
                        encountered.append(rec_itemnos[subj][serial_pos])
        result[subject_index] = count / lists if lists > 0 else np.nan

    return result


def avg_eli(intrusions=None, subjects=None):
    """
    A modification of the behavioral toolbox's xli function. Calculate's each partcipant's average number of ELIs per 
    list instead of their total number of ELIs.

    :param intrusions: An intrusions matrix in the format generated by recalls_to_intrusions
    :param subjects: A list of subject codes, indicating which subject produced each row of the intrusions matrix
    :return: An array where each entry is the average number of PLIs per list for a single participant.
    """
    usub = np.unique(subjects)
    result = np.zeros(len(usub))
    for subject_index in range(len(usub)):
        count = 0.
        lists = 0.
        for subj in range(len(subjects)):
            if subjects[subj] == usub[subject_index]:
                lists += 1
                for serial_pos in range(len(intrusions[0])):
                    if intrusions[subj][serial_pos] < 0:
                        count += 1
        result[subject_index] = count / lists if lists > 0 else np.nan
    return result


def avg_reps(rec_itemnos, subjects):
    """
    Calculate's each partcipant's average number of repetitions per list.
    
    :param rec_itemnos: A matrix in which each row is the list of IDs for all words recalled by a single subject on a
                        single trial. Rows are expected to be padded with 0s to all be the same length.
    :param subjects: A list of subject codes, indicating which subject produced each row of the intrusions matrix
    :return: An array where each entry is the average number of repetitions per list for a single participant.
    """
    usub = np.unique(subjects)
    result = np.zeros(len(usub))
    for subject_index in range(len(usub)):
        count = 0.
        lists = 0.
        for subj in range(len(subjects)):
            if subjects[subj] == usub[subject_index]:
                lists += 1
                # times_recalled is an array with one entry for each unique correctly recalled word, indicating the
                # number of times that word was recalled during the current list
                times_recalled = np.array([len(np.where(rec_itemnos[subj, :] == rec)[0]) for rec in np.unique(rec_itemnos[subj, :]) if rec > 0])
                # Subtract 1 from the number of times each correct word was recalled in the list to give the number of
                # repetitions
                repetitions = times_recalled - 1
                # Sum the number of repetitions made in the current list
                count += repetitions.sum()
        result[subject_index] = count / lists if lists > 0 else np.nan
    return result
