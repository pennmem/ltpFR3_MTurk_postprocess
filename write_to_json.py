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
        for stat in stats[subj]:
            for f in stats[subj][stat]:
                if isinstance(stats[subj][stat][f], np.ndarray):
                    stats[subj][stat][f] = stats[subj][stat][f].tolist()

    with open(stat_json, 'w') as f:
        json.dump(stats, f)
