import os
import json


def lookup_and_acceptance_tool():
    """
    Load worker ID/subject ID map JSON file and enter ID numbers when prompted to print the subject/worker ID that
    corresponds to them. As a shortcut, you can enter just the integer of a participant's number, e.g. 42 to look up
    MTK0042. Also checks whether that participant's submission was accepted or rejected, and asks whether you wish to
    accept or reject any new participant's submission.

    Requires administrator privileges to access IDs.
    """
    ID_MAP = '/data/eeg/scalp/ltp/admin_docs/MTurk/ltpFR3_IDs.json'
    ACCEPTED_REJECTED = '/data/eeg/scalp/ltp/ltpFR3_MTurk/acceptance.json'

    ##########
    #
    # Load ID map and acceptance record
    #
    ##########
    try:
        with open(ID_MAP, 'r') as f:
            idmap = json.load(f)
    except IOError:
        raise Exception('Unable to load ID map. You may not have permission to access it.')

    if os.path.exists(ACCEPTED_REJECTED):
        with open(ACCEPTED_REJECTED, 'r') as f:
            acceptance = json.load(f)
    else:
        acceptance = {}

    ##########
    #
    # UI Loop
    #
    ##########
    while True:
        inp = input('Please enter an ID or leave blank to exit: ')
        if inp == '':
            break
        if inp.isnumeric() and len(inp) <= 4:
            inp = 'MTK%04d' % int(inp)

        # Lookup input ID
        if inp not in idmap:
            print('ID not recognized!')
            continue
        print(idmap[inp])

        # Determine the subject ID (should be of the form "MTK####"). We only use subject IDs in the acceptance record.
        if inp.startswith('MTK') and len(inp) == 7:
            subjid = inp
        elif idmap[inp].startswith('MTK') and len(idmap[inp]) == 7:
            subjid = idmap[inp]
        else:
            continue

        # Use the subject ID to lookup whether the participant has been accepted or rejected
        if subjid in acceptance:
            if acceptance[subjid]:
                print('ACCEPTED')
            else:
                print('REJECTED')
            continue

        # If the participant has not yet been accepted or rejected, prompt user to mark as accepted or rejected
        while True:
            accept = input('Enter 1 to ACCEPT, 0 to REJECT, or leave blank to POSTPONE: ')
            if accept == '0':
                acceptance[subjid] = 0
                break
            elif accept == '1':
                acceptance[subjid] = 1
                break
            elif accept == '':
                break
            else:
                print('Invalid input!')

    # Save acceptance record
    print('Saving...')
    with open(ACCEPTED_REJECTED, 'w') as f:
        json.dump(acceptance, f)
    print('Done.')


if __name__ == "__main__":
    lookup_and_acceptance_tool()
