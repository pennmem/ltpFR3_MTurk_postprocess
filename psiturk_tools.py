from sqlalchemy import create_engine, MetaData, Table
import json
import numpy as np
import pandas as pd
from pyxdameraulevenshtein import damerau_levenshtein_distance_ndarray


def load_psiturk_data(db_url=None, table_name=None, data_column_name='datastring'):
    """
    Loads the data from a psiTurk experiment. Adapted from the official psiTurk docs.

    :param db_url: The location of the database as a string. If using mySQL, must include username and password, e.g.
    'mysql://user:password@127.0.0.1:3306/mturk_db'
    :param table_name: The name of the experiment's table within the database, e.g. 'ltpFR3'
    :param data_column_name: The name of the column in which psiTurk has stored the actual experiment event data. By
    default, psiTurk labels this column as 'datastring'
    :return: A pandas data frame with the data from column data_column_name, in table table_name, within database db_url
    """
    # boilerplate sqlalchemy setup
    engine = create_engine(db_url)
    metadata = MetaData()
    metadata.bind = engine
    table = Table(table_name, metadata, autoload=True)
    # make a query and loop through
    s = table.select()
    rows = s.execute()

    data = []
    # status codes of subjects who completed experiment
    statuses = [3, 4, 5, 7]
    # if you have workers you wish to exclude, add them here
    exclude = []
    for row in rows:
        # only use subjects who completed experiment and aren't excluded
        if row['status'] in statuses and row['uniqueid'] not in exclude:
            data.append(row[data_column_name])

    # Now we have all participant datastrings in a list.
    # Let's make it a bit easier to work with:

    # parse each participant's datastring as json object
    # and take the 'data' sub-object
    data = [json.loads(part)['data'] for part in data]

    # insert uniqueid field into trialdata in case it wasn't added
    # in experiment:
    for part in data:
        for record in part:
            record['trialdata']['uniqueid'] = record['uniqueid']

    # flatten nested list so we just have a list of the trialdata recorded
    # each time psiturk.recordTrialData(trialdata) was called.
    data = [record['trialdata'] for part in data for record in part]

    # Put all subjects' trial data into a dataframe object from the
    # 'pandas' python library: one option among many for analysis
    data_frame = pd.DataFrame(data)
    return data_frame


def process_psiturk_data(data, dict_path):
    """
    Post-process the raw psiTurk data extracted by load_psiturk_data. This involves creating recalls and presentation
    matrices, extracting list conditions, etc.

    :param data: The raw data frame built by load_psiturk_data out of the psiTurk server's SQL database.
    :param dict_path: The path to the text file (e.g. Webster's Dictionary) that will be used for spellchecking and
    looking up ELIs
    :return: A data dictionary, with one key for each participant. Each participant's number maps to a dictionary
    containing their recalls matrix, info on the conditions of each list, etc.
    """

    d = {}

    # Load the Webster's dictionary file, remove spaces, and make all words lowercase
    with open(dict_path, 'r') as df:
        dictionary = df.readlines()
    dictionary = [word.lower().strip() for word in dictionary if ' ' not in word]

    # Set filters for recall and presentation events
    recalls_filter = data.type == 'FREE_RECALL'
    study_filter_aud = data.type == 'PRES_AUD'
    study_filter_vis = data.type == 'PRES_VIS'

    # For each subject
    subjects = data.uniqueid.unique()
    for s in subjects:
        # Initialize data entries for subject
        d[s] = {}
        d[s]['recalls'] = []
        d[s]['rec_words'] = []
        d[s]['pres_words'] = []
        d[s]['list_len'] = []
        d[s]['pres_rate'] = []
        d[s]['pres_mod'] = []
        d[s]['dist_dur'] = []

        # Get all presentation and recall events from the current subject
        s_filter = data.uniqueid == s
        s_pres = data.loc[s_filter & study_filter_aud | study_filter_vis, ['trial', 'word', 'conditions']].as_matrix()
        s_recalls = data.loc[s_filter & recalls_filter, ['trial', 'recwords', 'conditions', 'rt']].as_matrix()
        pres_trials = np.array([x[0] for x in s_pres])
        pres_words = np.array([str(x[1]) for x in s_pres])
        rec_trials = np.array([x[0] for x in s_recalls])
        rec_words = np.array([[str(y) for y in x[1]] for x in s_recalls])

        # Add presented words to the data structure
        d[s]['pres_words'] = [[word for i, word in enumerate(pres_words) if pres_trials[i] == trial] for trial in np.unique(pres_trials)]

        # Get conditions for each trial and add them to the data structure
        conditions = [x[2] for x in s_recalls]
        d[s]['list_len'] = [x[0] for x in conditions]
        d[s]['pres_rate'] = [x[1] for x in conditions]
        d[s]['pres_mod'] = [x[2] for x in conditions]
        d[s]['dist_dur'] = [x[3] for x in conditions]

        # Create empty was_recalled matrix
        d[s]['was_recalled'] = np.zeros((len(d[s]['pres_words']), np.max(d[s]['list_len'])))

        # Add recall timing matrix to the data structure
        d[s]['rt'] = [x[3] for x in s_recalls]

        # For each trial in a subject's session
        for i, t in enumerate(np.unique(pres_trials)):
            # Get a list of all words presented so far this session (to be able to search for PLIs)
            presented_so_far = pres_words[np.where(pres_trials <= t)]
            # Get a list of the trials each word was presented on
            when_presented = pres_trials[np.where(pres_trials <= t)]
            # Get an array of the words recalled this trial
            recalled_this_list = rec_words[np.where(rec_trials == t)][0]

            # Mark each recall as a correct recall, ELI, PLI, or other and make a recall list for the current trial
            sp = []
            for recall in recalled_this_list:
                list_num, position = which_item(recall, t, presented_so_far, when_presented, dictionary)
                if list_num is None:
                    # ELIs and invalid strings get error code of -999 or -9999 listed in their recalls matrix
                    sp.append(position)
                else:
                    # Mark word as recalled
                    d[s]['was_recalled'][list_num, position-1] = 1
                    # PLIs get serial position of -n, where n is the number of lists back the word was presented
                    if list_num != t:
                        sp.append(list_num - t)
                    # Correct recalls get their serial position listed as is
                    else:
                        sp.append(position)

            # Add the current trial's recalls as a row in the participant's recalls matrix
            d[s]['recalls'].append(sp)
            d[s]['rec_words'].append(recalled_this_list)
            d[s]['was_recalled'][i, d[s]['list_len'][i]:] = np.nan

    return d


def which_item(recall, trial, presented, when_presented, dictionary):
    """
    Determine the serial position of a recalled word. Extra-list intrusions are identified by looking them up in a word
    list. Unrecognized words are spell-checked.

    :param recall: A string typed by the subject into the recall entry box
    :param trial: The trial number during which the recall was entered
    :param presented: The list of words seen by this subject so far, across all trials <= the current trial number
    :param when_presented: A listing of which trial each word was presented in
    :param dictionary: A list of strings that should be considered as possible extra-list intrusions
    :return: If a correct recall, the serial position of the recall. If a PLI, -n, where n is the number of lists back
    that the word was presented. If an ELI or invalid string, -999.
    """

    # Check whether the recall exactly matches a previously presented word
    seen, seen_where = self_term_search(recall, presented)

    # If word has been presented
    if seen:
        # Determine the list number and serial position of the word
        list_num = when_presented[seen_where]
        first_item = np.min(np.where(when_presented == list_num))
        serial_pos = seen_where - first_item + 1
        return int(list_num), int(serial_pos)

    # If the recalled word was not presented, but exactly matches any word in the dictionary, mark as an ELI
    in_dict, where_in_dict = self_term_search(recall, dictionary)
    if in_dict:
        return None, -999

    # If the recall contains non-letter characters
    if not recall.isalpha():
        return None, -999

    # If word is not in the dictionary, find the closest match based on edit distance
    recall = correct_spelling(recall, presented, dictionary)
    return which_item(recall, trial, presented, when_presented, dictionary)


def self_term_search(find_this, in_this):
    for index, word in enumerate(in_this):
        if word == find_this:
            return True, index
    return False, None


def correct_spelling(recall, presented, dictionary):

    # edit distance to each item in the pool and dictionary
    dist_to_pool = damerau_levenshtein_distance_ndarray(recall, np.array(presented))
    dist_to_dict = damerau_levenshtein_distance_ndarray(recall, np.array(dictionary))

    # position in distribution of dist_to_dict
    ptile = np.true_divide(sum(dist_to_dict <= np.amin(dist_to_pool)), dist_to_dict.size)

    # decide if it is a word in the pool or an ELI
    if ptile <= .1:
        corrected_recall = presented[np.argmin(dist_to_pool)]
    else:
        corrected_recall = dictionary[np.argmin(dist_to_dict)]
    recall = corrected_recall
    return recall
