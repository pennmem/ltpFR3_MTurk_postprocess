import csv
import json
from glob import glob

def process_survey(outfile):
    """

    :param outfile:
    :return:
    """
    # Load existing survey spreadsheet
    # Note that this assumes the data from early participants who took the separate questionnaire has been already saved
    with open(outfile, 'r') as f:
        r = csv.reader(f, delimiter=',')
        s = [row for row in r]
        head = s.pop(0)

    # Get locations of all session log files
    log_files_good = glob('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/MTK[0-9][0-9][0-9][0-9].json')
    log_files_excluded = glob('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/excluded/MTK[0-9][0-9][0-9][0-9].json')
    log_files_bad_sess = glob('/data/eeg/scalp/ltp/ltpFR3_MTurk/events/bad_sess/MTK[0-9][0-9][0-9][0-9].json')
    log_files_all = log_files_good + log_files_excluded + log_files_bad_sess

    # Get list of participants whose data has already been processed
    already_processed = [row[0] for row in s]

    # Process the responses from each new log file
    for lf in log_files_all:
        subj = lf[-12:-5]
        if subj in already_processed and not force:
            continue
        print(subj)

        # Load questionnaire data for participant
        with open(lf, 'r') as f:
            data = json.load(f)['questiondata']

        # Extract relevant info
        age = data['age'] if 'age' in data else 'Not Reported'
        education = data['education'] if 'education' in data else 'Not Reported'
        ethnicity = data['ethnicity'] if 'ethnicity' in data else 'Not Reported'
        gender = data['gender'] if 'gender' in data else 'Not Reported'
        gender_other = data['gender_other'] if 'gender_other' in data else ''
        language = data['language'] if 'language' in data else 'Not Reported'
        marital = data['marital'] if 'marital' in data else 'Not Reported'
        origin = data['origin'] if 'origin' in data else 'Not Reported'
        race = '|'.join(data['race']) if 'race' in data else 'Not Reported'
        race_other = data['race_other'] if 'race_other' in data else ''

        # Add new row to spreadsheet
        s.append([subj, age, education, ethnicity, gender, gender_other, language,
                  marital, origin, race, race_other])

        # Write data out to file
        with open(outfile, 'w') as f:
            w = csv.writer(f, delimiter=',')
            w.writerow(head)
            for row in s:
                w.writerow(row)
