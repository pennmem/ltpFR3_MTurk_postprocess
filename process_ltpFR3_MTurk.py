import psiturk_tools
from run_stats import run_stats
from report import ltpFR3_report

# Set paths
db_url = 'sqlite:////Users/jessepazdera/AtomProjects/ltpFR3-MTurk/participants.db'  # url for the database in which raw psiturk ouput is stored
table_name = 'ltpFR3'  # table of the database
dict_path = 'webster_dictionary.txt'  # dictionary to use when looking for ELIs and correcting spelling
save_file = 'ltpFR3.data'
exp_data_dir = ''  # path to data dir on circ2

# Load the data from the psiTurk experiment database and process it into a dictionary
data = psiturk_tools.load_psiturk_data(db_url, table_name)
data = psiturk_tools.process_psiturk_data(data, dict_path)

# Run stats on the data and create a PDF report for each participant
stats = run_stats(data)
ltpFR3_report(stats)
