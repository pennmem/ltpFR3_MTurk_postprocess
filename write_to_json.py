import json
import numpy as np


def write_to_json(data, stats, behmat_json, stat_json):
    """
    Converts all numpy arrays in data and stats to lists, then writes both objects to JSON files.

    :param data: A dictionary containing all ltpFR3 behavioral data matrices.
    :param stats: A dictionary containing all ltpFR3 behavioral performance stats.
    :param behmat_json: The path where data will be written.
    :param stat_json: The path where stats will be written
    """
    for subj in data:
        for field in data[subj]:
            if isinstance(data[subj][field], np.ndarray):
                data[subj][field] = data[subj][field].tolist()

    with open(behmat_json, 'w') as f:
        json.dump(data, f)

    for subj in stats:
        if subj == 'all':
            for stat in stats['all']['mean']:
                for f in stats['all']['mean'][stat]:
                    if isinstance(stats['all']['mean'][stat][f], np.ndarray):
                        stats['all']['mean'][stat][f] = stats['all']['mean'][stat][f].tolist()
            for stat in stats['all']['sem']:
                for f in stats['all']['sem'][stat]:
                    if isinstance(stats['all']['sem'][stat][f], np.ndarray):
                        stats['all']['sem'][stat][f] = stats['all']['sem'][stat][f].tolist()
        for stat in stats[subj]:
            if stat in ('rec_per_trial', 'math_per_trial'):
                stats[subj][stat] = stats[subj][stat].tolist()
            else:
                for f in stats[subj][stat]:
                    if isinstance(stats[subj][stat][f], np.ndarray):
                        stats[subj][stat][f] = stats[subj][stat][f].tolist()

    with open(stat_json, 'w') as f:
        json.dump(stats, f)
