import os
import json
import numpy as np
from glob import glob
from pybeh.spc import spc
from pybeh.pnr import pnr
from pybeh.crp import crp
from pybeh.xli import xli
from pybeh.reps import reps
from scipy.stats import sem
from pybeh.temp_fact import temp_fact
from pybeh.dist_fact import dist_fact
from write_to_json import write_stats_to_json


def run_stats(data_dir, stat_dir, force=False):
    """
    Use the data extracted from a psiTurk database to generate behavioral stats for each participant. Creates a stat
    data structure with the following format for each participant, then saves it to a JSON file:

    {
    stat1: {condition1: value, condition2: value, ...},
    stat2: {condition1: value, condition2: value, ...},
    stat3: {condition1: value, condition2: value, ...},
    ...
    }
    
    :param data_dir: The path to the directory where behavioral matrix data files are stored.
    :param report_dir: The path to the directory where new stats files will be saved.
    :param force: If False, only calculate stats for participants who do not already have a stats file (plus the average
     stat file). If True, calculate stats for all participants. (Default == False)
    """
    exclude = [s.decode('UTF-8') for s in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/EXCLUDED.txt', dtype='S8')]
    bad_sess = [s.decode('UTF-8') for s in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/BAD_SESS.txt', dtype='S8')]
    rejected = [s.decode('UTF-8') for s in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/REJECTED.txt', dtype='S8')]
    skip = np.union1d(np.union1d(exclude, bad_sess), rejected)
    with open('/data/eeg/scalp/ltp/ltpFR3_MTurk/VERSION_STARTS.json') as f:
        version_starts = json.load(f)
        version_starts = {int(v): version_starts[v] for v in version_starts}
    
    # Load wordpool and word2vec similarity
    wordpool = np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/wasnorm_wordpool.txt')
    w2v = np.loadtxt('/data/eeg/scalp/ltp/w2v.txt')

    filters = {
        # Grand average
        'all': {'ll': None, 'pr': None, 'mod': None, 'dd': None},

        # Modality
        'a': {'mod': 'a'}, 'v': {'mod': 'v'},

        # Pres Rate
        's': {'pr': 1600}, 'f': {'pr': 800},

        # List Length
        '12': {'ll': 12}, '24': {'ll': 24},

        # Distractor Duration
        'd': {'dd': 12000}, 'D': {'dd': 24000},

        # Modality x List Length
        'a12': {'ll': 12, 'mod': 'a'}, 'a24': {'ll': 24, 'mod': 'a'},
        'v12': {'ll': 12, 'mod': 'v'}, 'v24': {'ll': 24, 'mod': 'v'},

        # Pres Rate x List Length
        'f12': {'ll': 12, 'pr': 800}, 'f24': {'ll': 24, 'pr': 800},
        's12': {'ll': 12, 'pr': 1600}, 's24': {'ll': 24, 'pr': 1600},

        # Distractor Duration x List Length
        '12d': {'ll': 12, 'dd': 12000}, '24d': {'ll': 24, 'dd': 12000},
        '12D': {'ll': 12, 'dd': 24000}, '24D': {'ll': 24, 'dd': 24000},

        # Modality x Pres Rate x List Length
        'sa12': {'ll': 12, 'pr': 1600, 'mod': 'a'}, 'sa24': {'ll': 24, 'pr': 1600, 'mod': 'a'},
        'sv12': {'ll': 12, 'pr': 1600, 'mod': 'v'}, 'sv24': {'ll': 24, 'pr': 1600, 'mod': 'v'},
        'fa12': {'ll': 12, 'pr': 800, 'mod': 'a'}, 'fa24': {'ll': 24, 'pr': 800, 'mod': 'a'},
        'fv12': {'ll': 12, 'pr': 800, 'mod': 'v'}, 'fv24': {'ll': 24, 'pr': 800, 'mod': 'v'},

        # Modality x Distractor Duration x List Length
        'a12d': {'ll': 12, 'dd': 12000, 'mod': 'a'}, 'a24d': {'ll': 24, 'dd': 12000, 'mod': 'a'},
        'v12d': {'ll': 12, 'dd': 12000, 'mod': 'v'}, 'v24d': {'ll': 24, 'dd': 12000, 'mod': 'v'},
        'a12D': {'ll': 12, 'dd': 24000, 'mod': 'a'}, 'a24D': {'ll': 24, 'dd': 24000, 'mod': 'a'},
        'v12D': {'ll': 12, 'dd': 24000, 'mod': 'v'}, 'v24D': {'ll': 24, 'dd': 24000, 'mod': 'v'},

        # Pres Rate x Distractor Duration x List Length
        's12d': {'ll': 12, 'pr': 1600, 'dd': 12000}, 's24d': {'ll': 24, 'pr': 1600, 'dd': 12000},
        's12D': {'ll': 12, 'pr': 1600, 'dd': 24000}, 's24D': {'ll': 24, 'pr': 1600, 'dd': 24000},
        'f12d': {'ll': 12, 'pr': 800, 'dd': 12000}, 'f24d': {'ll': 24, 'pr': 800, 'dd': 12000},
        'f12D': {'ll': 12, 'pr': 800, 'dd': 24000}, 'f24D': {'ll': 24, 'pr': 800, 'dd': 24000},

        # Modality x Pres Rate x Distractor Duration x List Length
        'sa12d': {'ll': 12, 'pr': 1600, 'mod': 'a', 'dd': 12000}, 'sa24d': {'ll': 24, 'pr': 1600, 'mod': 'a', 'dd': 12000},
        'sv12d': {'ll': 12, 'pr': 1600, 'mod': 'v', 'dd': 12000}, 'sv24d': {'ll': 24, 'pr': 1600, 'mod': 'v', 'dd': 12000},
        'fa12d': {'ll': 12, 'pr': 800, 'mod': 'a', 'dd': 12000}, 'fa24d': {'ll': 24, 'pr': 800, 'mod': 'a', 'dd': 12000},
        'fv12d': {'ll': 12, 'pr': 800, 'mod': 'v', 'dd': 12000}, 'fv24d': {'ll': 24, 'pr': 800, 'mod': 'v', 'dd': 12000},
        'sa12D': {'ll': 12, 'pr': 1600, 'mod': 'a', 'dd': 24000}, 'sa24D': {'ll': 24, 'pr': 1600, 'mod': 'a', 'dd': 24000},
        'sv12D': {'ll': 12, 'pr': 1600, 'mod': 'v', 'dd': 24000}, 'sv24D': {'ll': 24, 'pr': 1600, 'mod': 'v', 'dd': 24000},
        'fa12D': {'ll': 12, 'pr': 800, 'mod': 'a', 'dd': 24000}, 'fa24D': {'ll': 24, 'pr': 800, 'mod': 'a', 'dd': 24000},
        'fv12D': {'ll': 12, 'pr': 800, 'mod': 'v', 'dd': 24000}, 'fv24D': {'ll': 24, 'pr': 800, 'mod': 'v', 'dd': 24000}
    }

    # Calculate stats for new participants
    for data_file in glob(os.path.join(data_dir, '*.json')):
        subj = os.path.splitext(os.path.basename(data_file))[0]  # Get subject ID from file name
        outfile = os.path.join(stat_dir, '%s.json' % subj)  # Define file path for stat file
        if subj in bad_sess or ((os.path.exists(outfile) or subj in exclude or subj in rejected) and not force):  # Skip participants who already had stats calculated
            continue

        with open(data_file, 'r') as f:
            d = json.load(f)

        # Extract behavioral matrices from JSON object and convert to numpy arrays; drop practice trials
        sub = np.array(d['subject'])[2:]
        condi = [(d['list_len'][i], d['pres_rate'][i], str(d['pres_mod'][i]), d['dist_dur'][i]) for i in range(len(d['serialpos']))][2:]
        recalls = np.array(d['serialpos'])[2:]
        wasrec = np.array(d['recalled'])[2:]
        ffr_wasrec = np.array(d['ffr_recalled'])[2:]
        rt = np.array(d['rt'])[2:]
        recw = np.array(d['rec_words'])[2:]
        presw = np.array(d['pres_words'])[2:]
        recnos = np.searchsorted(wordpool, recw, side='right')
        presnos = np.searchsorted(wordpool, presw, side='right')
        intru = np.array(d['intrusions'])[2:]
        math = np.array(d['math_correct'])[2:]

        # Run all stats for a single participant and add the resulting stats object to the stats dictionary
        stats = stats_for_subj(sub, condi, recalls, wasrec, ffr_wasrec, rt, recw, presw, recnos, presnos, intru, math, filters, w2v)
        write_stats_to_json(stats, outfile)

    # Calculate average stats. First we need to load the stats from all participants.
    stats = {}
    for stat_file in glob(os.path.join(stat_dir, '*.json')):
        subj = os.path.splitext(os.path.basename(stat_file))[0]
        if not subj.startswith('all'):
            with open(stat_file, 'r') as f:
                stats[subj] = json.load(f)

    # Now we calculate the average stats and save to a JSON file for each experiment
    for version in version_starts:
        avg_stats = {}
        v_start = version_starts[version]
        v_end = None if version + 1 not in version_starts else version_starts[version + 1]

        outfile = os.path.join(stat_dir, 'all_v%d.json' % version)
        avg_stats['mean'], avg_stats['sem'], avg_stats['N'] = calculate_avg_stats(stats, filters.keys(),
                                                                                  version_start=v_start,
                                                                                  version_end=v_end,
                                                                                  exclude_wrote_notes=False)
        write_stats_to_json(avg_stats, outfile, average_stats=True)

        outfile = os.path.join(stat_dir, 'all_v%d_excl_wn.json' % version)
        avg_stats['mean'], avg_stats['sem'], avg_stats['N'] = calculate_avg_stats(stats, filters.keys(),
                                                                                  version_start=v_start,
                                                                                  version_end=v_end,
                                                                                  exclude_wrote_notes=True)
        write_stats_to_json(avg_stats, outfile, average_stats=True)


def stats_for_subj(sub, condi, recalls, wasrec, ffr_wasrec, rt, recw, presw, recnos, presnos, intru, math, filters, w2v):
    """
    Create a stats dictionary for a single participant.
    
    :param sub: A subject array for the stats calculations. As we are only running stats on one participant at a time, 
    this array will simply be [subj_name] * number_of_trials.
    :param condi: An array of tuples indicating which conditions were used on each trial (ll, pr, mod, dd).
    :param recalls: A matrix where item (i, j) is the serial position of the ith recall in trial j.
    :param wasrec: A matrix where item (i, j) is 0 if the ith presented word in trial j was not recalled, 1 if it was.
    :param ffr_wasrec: A matrix where item (i, j) is 0 if the ith presented word in trial j was not recalled during FFR,
            1 if it was.
    :param rt: A matrix where item (i, j) is the response time (in ms) of the ith recall in trial j.
    :param recw: A matrix where item (i, j) is the ith word recalled on trial j.
    :param presw: A matrix where item (i, j) is the ith word presented on trial j.
    :param intru: A list x items intrusions matrix (see recalls_to_intrusions).
    :return: 
    """
    stats_to_run = ['spc', 'ffr_spc', 'pfr', 'psr', 'ptr', 'crp_early', 'crp_late', 'temp_fact', 'irt',
                    'spc_fr1', 'spc_frl4', 'irt_sp_excl', 'prec', 'pffr', 'pffr_rec', 'pffr_unrec',
                    'elis', 'reps', 'pli_recency', 'plis', 'pli_recency_2factor', 'plis_2factor',
                    'rec_per_trial', 'math_per_trial']
    stats = {stat: {} for stat in stats_to_run}
    for f in filters:
        # Get presentation and recall info just from trials that match the filter's set of conditions
        fsub, frecalls, fwasrec, fffr_wasrec, frt, frecw, fpresw, frecnos, fpresnos, fintru = [filter_by_condi(a, condi, **filters[f]) for a in [sub, recalls, wasrec, ffr_wasrec, rt, recw, presw, recnos, presnos, intru]]

        # If no trials match the current filter, skip to next filter
        if fsub is None:
            continue

        # Calculate stats on all lists within the current condition. Note that some stats require a single list length.
        ll = filters[f]['ll'] if 'll' in filters[f] else None
        if ll is not None:
            # SPC, PFR/PSR/PTR, CRP, TemF, IRT
            stats['spc'][f] = spc(frecalls, fsub, ll)[0]
            stats['ffr_spc'][f] = ffr_spc(fffr_wasrec, fsub, ll)[0]
            stats['pfr'][f] = pnr(frecalls, fsub, ll, n=0)[0]
            stats['psr'][f] = pnr(frecalls, fsub, ll, n=1)[0]
            stats['ptr'][f] = pnr(frecalls, fsub, ll, n=2)[0]
            stats['crp_early'][f] = crp(frecalls[:, :3], fsub, ll, lag_num=5)[0]
            stats['crp_late'][f] = crp(frecalls, fsub, ll, lag_num=5, skip_first_n=2)[0]
            stats['temp_fact'][f] = temp_fact(frecalls, fsub, ll, skip_first_n=2)[0]
            stats['sem_fact'][f] = dist_fact(frecnos, fpresnos, fsub, w2v, skip_first_n=2)[0]
            stats['irt'][f] = irt_subj(frt, frecalls, ll)

            # SPCs by start position
            start1_mask = frecalls[:, 0] == 1
            startl4_mask = frecalls[:, 0] > ll - 4
            if np.any(start1_mask):
                stats['spc_fr1'][f] = spc(frecalls[start1_mask, :], fsub[start1_mask], ll)[0]
            else:
                stats['spc_fr1'][f] = np.empty(ll)
                stats['spc_fr1'][f].fill(np.nan)
            if np.any(startl4_mask):
                stats['spc_frl4'][f] = spc(frecalls[startl4_mask, :], fsub[startl4_mask], ll)[0]
            else:
                stats['spc_frl4'][f] = np.empty(ll)
                stats['spc_frl4'][f].fill(np.nan)

            # Special version of the IRT which excludes trials that had a recall within the last 10 seconds
            rt_exclusion_mask = np.max(frt, axis=1) <= 50000
            if rt_exclusion_mask.sum() > 0:
                stats['irt_sp_excl'][f] = irt_subj(frt[rt_exclusion_mask], frecalls[rt_exclusion_mask], ll)
            else:
                stats['irt_sp_excl'][f] = np.empty((ll+1, ll+1))
                stats['irt_sp_excl'][f].fill(np.nan)

        # PRec, PFFR, Intrusions
        stats['prec'][f] = prec(fwasrec, fsub)[0]
        stats['pffr'][f], stats['pffr_rec'][f], stats['pffr_unrec'][f] = pffr_subj(fffr_wasrec, fpresw, frecw)
        stats['elis'][f] = xli(fintru, fsub, per_list=True)[0]
        stats['reps'][f] = reps(frecalls, fsub, per_list=True)[0]
        stats['plis'][f] = plis_1factor(fintru, n_skip=1)
        stats['pli_recency'][f] = plir_1factor(intru, condi, n_points=5, n_skip=2, **filters[f])

    # PLIs and PLI recency (2 x 2 modality) - Experiment 1 only (i.e., MTK001 -- MTK1308)
    if int(sub[0][-4:]) <= 1308:
        mods = np.array([c[2] for c in condi])
        stats['plis_2factor'] = plis_2factor(intru, mods)
        stats['pli_recency_2factor'] = plir_2factor(intru, mods, n_points=5, n_skip=2)
    else:
        del stats['plis_2factor']
        del stats['pli_recency_2factor']

    # PRec and math performance by trial
    stats['rec_per_trial'] = np.nanmean(wasrec, axis=1)
    stats['math_per_trial'] = np.sum(math, axis=1)

    return stats


def calculate_avg_stats(s, filters, version_start=1, version_end=None, exclude_wrote_notes=False):
    stats_to_run = ['spc', 'ffr_spc', 'pfr', 'psr', 'ptr', 'crp_early', 'crp_late', 'temp_fact', 'sem_fact', 'irt',
                    'spc_fr1', 'spc_frl4', 'irt_sp_excl', 'prec', 'pffr', 'pffr_rec', 'pffr_unrec',
                    'elis', 'reps', 'pli_recency', 'plis', 'pli_recency_2factor', 'plis_2factor',
                    'rec_per_trial', 'math_per_trial']

    # Exclusion notes can be found at: https://app.asana.com/0/291595828487527/468440625589939/f
    exclude = [b.decode('UTF-8') for b in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/EXCLUDED.txt', dtype='S8')]
    bad_sess = [b.decode('UTF-8') for b in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/BAD_SESS.txt', dtype='S8')]
    rejected = [b.decode('UTF-8') for b in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/REJECTED.txt', dtype='S8')]
    skip = np.union1d(np.union1d(exclude, bad_sess), rejected)

    if exclude_wrote_notes:
        wrote_notes = [b.decode('UTF-8') for b in np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/WROTE_NOTES.txt', dtype='S8')]
        skip = np.union1d(skip, wrote_notes)

    avs = {}
    stderr = {}
    Ns = {}
    for stat in stats_to_run:

        # Stats without filters
        if stat in ('plis_2factor', 'pli_recency_2factor', 'rec_per_trial', 'math_per_trial'):

            # Get the scores from all subjects from the target experiment who are not excluded
            scores = []
            for subj in s:
                snum = int(subj[-4:])
                if (subj not in skip) and (version_start is None or snum >= version_start) and \
                        (version_end is None or snum < version_end) and stat in s[subj]:
                    scores.append(s[subj][stat])
            scores = np.array(scores)

            # Average across subjects
            avs[stat] = np.nanmean(scores, axis=0)
            stderr[stat] = sem(scores, axis=0, nan_policy='omit')
            Ns[stat] = np.sum(np.logical_not(np.isnan(scores)), axis=0, dtype=np.float64)

        # Stats with filters
        else:
            avs[stat] = {}
            stderr[stat] = {}
            Ns[stat] = {}
            for f in filters:
                # For the "all" filter, skip any stat that requires a fixed list length (e.g. SPC)
                if f == 'all' and stat not in ('prec', 'pffr', 'pffr_rec', 'pffr_unrec', 'plis', 'elis', 'reps', 'pli_recency'):
                    continue

                # Get the scores from all subjects from the target experiment who are not excluded
                scores = []
                for subj in s:
                    snum = int(subj[-4:])
                    if (subj not in skip) and (version_start is None or snum >= version_start) and \
                            (version_end is None or snum < version_end) and f in s[subj][stat]:
                        scores.append(s[subj][stat][f])
                scores = np.array(scores)

                # Average across subjects
                avs[stat][f] = np.nanmean(scores, axis=0)
                stderr[stat][f] = sem(scores, axis=0, nan_policy='omit')
                Ns[stat][f] = np.sum(np.logical_not(np.isnan(scores)), axis=0, dtype=np.float64)

    return avs, stderr, Ns


def filter_by_condi(a, condi, ll=None, pr=None, mod=None, dd=None):
    """
    Filter a matrix of trial data to only get the data from trials that match the specified condition(s)
    :param a: A numpy array with the data from one trial on each row
    :param condi: A list of tuples indicating the list length, presentation rate, modality, distractor duration of each trial
    :param ll: Return only trials with this list length condition (ignore if None)
    :param pr: Return only trials with this presentation rate condition (ignore if None)
    :param mod: Return only trials with this presentation modality condition (ignore if None)
    :param dd: Return only trials with this distractor duration condition (ignore if None)
    :return: A numpy array containing only the data from trials that match the specified condition(s)
    """
    ind = [i for i in range(len(condi)) if ((ll is None or condi[i][0] == ll) and (pr is None or condi[i][1] == pr) and (mod is None or condi[i][2] == mod) and (dd is None or condi[i][3] == dd))]
    if len(ind) == 0:
        return None
    return a[ind]


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
    result = np.array([np.nanmean(recalled[subjects == subj]) for subj in usub])
    stderr = np.array([sem(recalled[subjects == subj], nan_policy='omit') for subj in usub])

    return result, stderr


def ffr_spc(ffr_recalled, subjects, ll):
    if len(ffr_recalled) == 0:
        return np.array([])
    usub = np.unique(subjects)
    result = np.array([np.nanmean(ffr_recalled[subjects == subj], axis=0) for subj in usub])
    return result[:, :ll]


def plir_1factor(intrusions, condi, n_points=5, n_skip=2, ll=None, pr=None, mod=None, dd=None):
    """
    Calculate the ratio of PLIs that originated from 1 list back, 2 lists back, etc. up until nmax lists back.

    :param intrusions: An intrusions matrix in the format generated by recalls_to_intrusions
    :param condi: A list of tuples indicating the list length, presentation rate, modality, distractor duration of each trial
    :param n_points: The maximum number of lists back to consider
    :param n_skip: The number of trials to skip at the beginning of the session.
    :param ll: Return only trials with this list length condition (ignore if None)
    :param pr: Return only trials with this presentation rate condition (ignore if None)
    :param mod: Return only trials with this presentation modality condition (ignore if None)
    :param dd: Return only trials with this distractor duration condition (ignore if None)
    :return: An array of length n_points, where item i is the ratio of PLIs that originated from i+1 lists back.
    """
    n_trials = len(intrusions)
    pli_recency = np.zeros((n_trials, n_trials))  # trial x recency
    condi_mask = np.array([True if (
            (ll is None or condi[i][0] == ll) and
            (pr is None or condi[i][1] == pr) and
            (mod is None or condi[i][2] == mod) and
            (dd is None or condi[i][3] == dd)
    ) else False for i in range(len(condi))], dtype=bool)

    for trial, trial_data in enumerate(intrusions):

        if trial < n_skip or not condi_mask[trial]:
            pli_recency[trial, :].fill(np.nan)
        else:
            # Mark NaNs for all lags that are not possible on the current trial
            # i.e. the third trial can only have PLIs of lag -1 and -2
            pli_recency[trial, trial:].fill(np.nan)
            # Count PLIs at lags that were possible
            trial_plis = trial_data[(trial_data > 0) & (trial_data <= trial)]
            for pli in trial_plis:
                pli_recency[trial, pli-1] += 1

    # Average PLIs across trials at each recency
    pli_recency = np.nanmean(pli_recency, axis=0)

    # Convert PLI averages to proportions of PLIs in each condition
    plis = np.nansum(pli_recency)
    pli_recency = pli_recency / plis if plis != 0 else np.full(n_trials, np.nan)

    # Extract just the recency score for n lists back
    pli_recency = pli_recency[:n_points]

    return pli_recency


def plir_2factor(intrusions, modalities, n_points=5, n_skip=2):
    """
    Calculate the ratio of PLIs that originated from 1 list back, 2 lists back, etc. up until nmax lists back.

    :param intrusions: An intrusions matrix in the format generated by recalls_to_intrusions
    :param modalities: A list of strings indicating the modality of each trial
    :param n_points: The maximum number of lists back to consider
    :param n_skip: The number of trials to skip at the beginning of the session.
    :return: An array of length n_points, where item i is the ratio of PLIs expected to originate from i+1 lists back
    """
    n_trials = len(intrusions)
    pli_recency = np.zeros((n_trials, n_trials, 2, 2))  # trial x recency x encoding modality x retrieval modality
    intru = intrusions
    mod = modalities

    for trial, trial_data in enumerate(intru):
        # Skip the first n trials
        if trial < n_skip:
            pli_recency[trial, :, :].fill(np.nan)
            continue

        # Get the modality of the current trial
        cur_mod = mod[trial]

        # Construct an array indicating the modality of the trial at each valid recency position
        # (most recent list at position 0, two lists back at position 1, etc. with NaNs for all impossible recencies)
        past_mods = mod.copy()
        past_mods[:trial] = past_mods[:trial][::-1]
        past_mods[trial:].fill(np.nan)

        # Mark NaNs in the cells that cannot have data for each trial. Visual data will go in modality index 0, auditory data will go in index 1.
        pli_recency[trial, past_mods != 'v', 0, :] = np.nan
        pli_recency[trial, past_mods != 'a', 1, :] = np.nan
        pli_recency[trial, :, :, int(cur_mod == 'v')] = np.nan

        # Use int(trial_mods[pli] == 'a') to send visual data into index 0 and auditory data into index 1
        trial_plis = trial_data[(trial_data > 0) & (trial_data <= trial)]
        for pli in trial_plis:
            pli_recency[trial, pli-1, int(past_mods[pli-1] == 'a'), int(cur_mod == 'a')] += 1

    # Average PLIs across trials at each recency and condition
    pli_recency = np.nanmean(pli_recency, axis=0)

    # Convert PLI averages to proportions of PLIs in each condition
    plis = np.nansum(pli_recency, axis=0)
    plis[plis == 0].fill(np.nan)
    pli_recency[:, 0, 0] /= plis[0, 0]
    pli_recency[:, 0, 1] /= plis[0, 1]
    pli_recency[:, 1, 0] /= plis[1, 0]
    pli_recency[:, 1, 1] /= plis[1, 1]

    # Extract just the recency scores for n_points list back
    pli_recency = pli_recency[:n_points, :, :]

    return pli_recency


def plis_1factor(intrusions, n_skip=1):
    """
    Calculate the average number of prior list intrusions per trial, while excluding PLI words that were presented
    on practice trials.

    :param intrusions:
    :param n_skip:
    :return:
    """
    plis = 0.
    possibles = 0.
    for trial, trial_data in enumerate(intrusions):
        if trial < n_skip:
            continue
        plis += np.logical_and(trial_data > 0, trial_data <= trial).sum()
        possibles += 1.
    plis = plis / possibles if possibles != 0 else np.nan
    return plis


def plis_2factor(intrusions, modalities):
    """
    Calculate the average number of prior list intrusions by encoding and retrieval modality, given that it is possible
    to make such a PLI.

    :param intrusions:
    :param modalities:
    :param n_skip:
    :return:
    """
    plis = np.zeros((2, 2))
    possibles = np.zeros((2, 2))

    had_v_trial = False
    had_a_trial = False
    for trial, trial_data in enumerate(intrusions):

        # Determine modality of current trial
        ret_mod = modalities[trial]
        ret_mod_is_aud = int(ret_mod == 'a')

        # Determine which types of PLIs are possible based on current list modality and whether the participant
        # previously had at least one visual and one auditory list
        if had_v_trial:
            possibles[0, ret_mod_is_aud] += 1
        if had_a_trial:
            possibles[1, ret_mod_is_aud] += 1

        # Determine the encoding modality of each PLI and add it to the appropriate count
        trial_plis = trial_data[(trial_data > 0) & (trial_data <= trial)]
        for pli in trial_plis:
            enc_mod = modalities[trial - pli]
            enc_mod_is_aud = int(enc_mod == 'a')
            plis[enc_mod_is_aud, ret_mod_is_aud] += 1

        # Once the participant has completed at least one trial of a given modality, mark had_v/a_trial as True,
        # as this influences which PLI types are possible
        if ret_mod_is_aud:
            had_a_trial = True
        else:
            had_v_trial = True

    # Convert to PLIs per list. Only count lists on which it was actually possible to make an intrusion of a given type.
    possibles[possibles == 0].fill(np.nan)
    plis /= possibles

    return plis


def irt_subj(rectimes, recalls, ll):
    irt = np.zeros((ll+1, ll+1))
    trial_count = np.zeros_like(irt)

    for trial, row in enumerate(rectimes):
        # Get the first output position at which each word was recalled, filtering out intrusions and repetitions
        spos, idx = np.unique(recalls[trial], return_index=True)
        idx = idx[spos > 0]

        # Get only the RTs for correct recalls
        masked_rt = row[np.sort(idx)]

        # Count the number of correct recalls
        num_recs = np.sum(masked_rt != 0)

        # Group trials' RTs based on the number of correct recalls made
        if num_recs > 1:
            irt[num_recs, :num_recs - 1] += np.diff(masked_rt)
            trial_count[num_recs, :] += 1

    # Set NaNs where no data is available
    irt[irt == 0] = np.nan
    irt = irt / trial_count

    return irt


def pffr_subj(ffr_rec, pres_words, rec_words):

    # Construct a special version of the recalled matrix, which indicates whether each presented word was *ever*
    # recalled (either correctly or as an intrusion)
    rec_words = rec_words.flatten()
    prev_rec = np.full(pres_words.shape, np.nan)
    pad_mask = pres_words != '0'
    for trial in range(ffr_rec.shape[0]):
        prev_rec[trial, pad_mask[trial]] = np.in1d(pres_words[trial, pad_mask[trial]], rec_words)

    # Probability of final free recall for all words
    overall_pffr = np.nanmean(ffr_rec)
    # Probability of final free recall for previously recalled words
    prev_rec_pffr = np.nanmean(ffr_rec[prev_rec == 1])
    # Probability of final free recall for previously unrecalled words
    prev_unrec_pffr = np.nanmean(ffr_rec[prev_rec != 1])

    return overall_pffr, prev_rec_pffr, prev_unrec_pffr
