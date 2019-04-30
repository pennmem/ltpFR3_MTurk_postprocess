import json
import numpy as np


def write_data_to_json(data, outfile):
    """
    Makes a behavioral matrix dictionary JSON-serializable, then saves it to a JSON file.
    :param data: A behavioral matrix dictionary for one participant.
    :param outfile: The file path where the data will be saved.
    """
    for field in data:
        if isinstance(data[field], np.ndarray):
            data[field] = data[field].tolist()

    with open(outfile, 'w') as f:
        json.dump(data, f)


def write_stats_to_json(stats, outfile, average_stats=False):
    """
    Makes a stats dictionary JSON-serializable, then saves it to a JSON file.
    :param stats: A stats dictionary for one participant (or the average stats dictionary).
    :param outfile: The file path where the data will be saved.
    :param average_stats: If True, assume that stats dictionary contains one entry with means for each condition and one
    entry for standard errors for each condition. (Default == False)
    """
    if average_stats:
        for stat in stats['mean']:
            if isinstance(stats['mean'][stat], np.ndarray):
                stats['mean'][stat] = stats['mean'][stat].tolist()
            elif isinstance(stats['mean'][stat], dict):
                for f in stats['mean'][stat]:
                    if isinstance(stats['mean'][stat][f], np.ndarray):
                        stats['mean'][stat][f] = stats['mean'][stat][f].tolist()
        for stat in stats['sem']:
            if isinstance(stats['sem'][stat], np.ndarray):
                stats['sem'][stat] = stats['sem'][stat].tolist()
            elif isinstance(stats['sem'][stat], dict):
                for f in stats['sem'][stat]:
                    if isinstance(stats['sem'][stat][f], np.ndarray):
                        stats['sem'][stat][f] = stats['sem'][stat][f].tolist()
        for stat in stats['N']:
            if isinstance(stats['N'][stat], np.ndarray):
                stats['N'][stat] = stats['N'][stat].tolist()
            elif isinstance(stats['N'][stat], dict):
                for f in stats['N'][stat]:
                    if isinstance(stats['N'][stat][f], np.ndarray):
                        stats['N'][stat][f] = stats['N'][stat][f].tolist()
    else:
        for stat in stats:
            if isinstance(stats[stat], np.ndarray):
                stats[stat] = stats[stat].tolist()
            elif isinstance(stats[stat], dict):
                for f in stats[stat]:
                    if isinstance(stats[stat][f], np.ndarray):
                        stats[stat][f] = stats[stat][f].tolist()

    with open(outfile, 'w') as f:
        json.dump(stats, f)
