import psiturk_tools
from run_stats import run_stats
from report import ltpFR3_report

# Set paths
db_url = 'sqlite:////data/eeg/scalp/ltp/ltpFR3_MTurk/ltpFR3_anonymized.db'  # url for the database in which raw psiturk ouput is stored
table_name = 'ltpFR3'  # table of the database
dict_path = 'webster_dictionary.txt'  # dictionary to use when looking for ELIs and correcting spelling
event_dir = '/data/eeg/scalp/ltp/ltpFR3_MTurk/events/'
behmat_dir = '/data/eeg/scalp/ltp/ltpFR3_MTurk/data/'
stat_dir = '/data/eeg/scalp/ltp/ltpFR3_MTurk/stats/'
report_dir = '/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/'

# Load the data from the psiTurk experiment database and process it into JSON files
psiturk_tools.load_psiturk_data(db_url=db_url, table_name=table_name, event_dir=event_dir)
# Create behavioral matrices from JSON data for each participant and save to JSON files
psiturk_tools.process_psiturk_data(event_dir, behmat_dir, dict_path)
# Run stats on the behavioral matrices from each participant and save to JSON files
run_stats(behmat_dir, stat_dir)
# Generate a PDF report for each participant, along with an aggregate report
ltpFR3_report(stat_dir, report_dir)

