import psiturk_tools
from run_stats import run_stats
from report import ltpFR3_report
from write_to_json import write_to_json

# Set paths
db_url = 'sqlite:////data/eeg/scalp/ltp/ltpFR3_MTurk/ltpFR3_anonymized.db'  # url for the database in which raw psiturk ouput is stored
table_name = 'ltpFR3'  # table of the database
dict_path = 'webster_dictionary.txt'  # dictionary to use when looking for ELIs and correcting spelling
save_file = 'ltpFR3.data'
behmat_json = '/data/eeg/scalp/ltp/ltpFR3_MTurk/ltpFR3_data.json'
stat_json = '/data/eeg/scalp/ltp/ltpFR3_MTurk/ltpFR3_stats.json'

# Load the data from the psiTurk experiment database and process it into a dictionary, then save to JSON
data = psiturk_tools.load_psiturk_data(db_url, table_name)
data = psiturk_tools.process_psiturk_data(data, dict_path)

# Run stats on the data and save to JSON
stats = run_stats(data)

# Generate a PDF report for each participant, along with an aggregate report
ltpFR3_report(stats)

write_to_json(data, stats, behmat_json, stat_json)
