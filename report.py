import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def ltpFR3_report(stats):
    """
    Generates a report from the input stats dictionary. The expected entries in the dictionary are:
    
    prec (Probability of recall)
    spc (Serial position curve)
    pfr (Probability of first recall)
    psr (Probability of second recall)
    ptr (Probability of third recall)
    crp_early (Conditional response probability among first three recalls)
    crp_late (Conditional response probability among recalls after the third)
    irt (Inter-item response time)
    pli_early (Average prior list intrusions per list among the first three recalls)
    pli_late (Average prior list intrusions per list among recalls after the third)
    eli_early (Average extra list intrusions per list among the first three recalls)
    eli_late (Average extra list intrusions per list among recalls after the third)
    reps (Average number of repetitions per list)
    pli_recency (Ratio of prior list intrusions coming from each list back, up to 6 back)
    
    And each entry contains a sub-dictionary with entries labelled 12, 18, and 24, which contain that stat for lists of
    length 12, 18, and 24, respectively. Note that length-18 lists are practice lists only.
    
    :param stats: A dictionary containing the behavioral stats calculated by run_stats.
    """
    stat_plotters = {'prec': plot_prec, 'spc': plot_spc, 'pfr': plot_pfr, 'psr': plot_psr, 'ptr': plot_ptr,
                      'crp_early': plot_crp_early, 'crp_late': plot_crp_late,
                      'pli_early': plot_pli_early, 'pli_late': plot_pli_late, 'eli_early': plot_eli_early,
                      'eli_late': plot_eli_late, 'reps': plot_reps, 'pli_recency': plot_pli_recency}

    for subj in stats:
        pdf = PdfPages('/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/' + subj + '.pdf')
        plt.figure(figsize=(40, 30))
        plt.suptitle(subj, fontsize=36)
        for key in stat_plotters:
            if key in stats[subj]:
                stat_plotters[key](stats[subj][key])
            else:
                print('ALERT! Missing stat %s for subject %s' % (key, subj))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig()
        pdf.close()
        plt.close()


def plot_spc(s):
    plt.subplot(8, 3, 1)
    plt.plot(range(1, 13), s['f12'])
    plt.plot(range(1, 25), s['f24'])
    plt.plot(range(1, 13), s['s12'])
    plt.plot(range(1, 25), s['s24'])
    plt.title('SPC')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Recall')
    plt.legend(labels=['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))

    plt.subplot(8, 3, 4)
    plt.plot(range(1, 13), s['v12'])
    plt.plot(range(1, 25), s['v24'])
    plt.plot(range(1, 13), s['a12'])
    plt.plot(range(1, 25), s['a24'])
    plt.title('SPC')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Recall')
    plt.legend(labels=['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_crp_early(s):
    plt.subplot(8, 3, 2)
    plt.plot(range(-3, 4), s['f12'])
    plt.plot(range(-3, 4), s['f24'])
    plt.plot(range(-3, 4), s['s12'])
    plt.plot(range(-3, 4), s['s24'])
    plt.title('CRP (Early)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])
    plt.ylim(0, 1)

    plt.subplot(8, 3, 5)
    plt.plot(range(-3, 4), s['v12'])
    plt.plot(range(-3, 4), s['v24'])
    plt.plot(range(-3, 4), s['a12'])
    plt.plot(range(-3, 4), s['a24'])
    plt.title('CRP (Early)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])
    plt.ylim(0, 1)


def plot_crp_late(s):
    plt.subplot(8, 3, 3)
    plt.plot(range(-3, 4), s['f12'])
    plt.plot(range(-3, 4), s['f24'])
    plt.plot(range(-3, 4), s['s12'])
    plt.plot(range(-3, 4), s['s24'])
    plt.title('CRP (Early)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])
    plt.ylim(0, 1)

    plt.subplot(8, 3, 6)
    plt.plot(range(-3, 4), s['v12'])
    plt.plot(range(-3, 4), s['v24'])
    plt.plot(range(-3, 4), s['a12'])
    plt.plot(range(-3, 4), s['a24'])
    plt.title('CRP (Late)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])
    plt.ylim(0, 1)


def plot_pfr(s):
    plt.subplot(8, 3, 7)
    plt.plot(range(1, 13), s['f12'])
    plt.plot(range(1, 25), s['f24'])
    plt.plot(range(1, 13), s['s12'])
    plt.plot(range(1, 25), s['s24'])
    plt.title('PFR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of First Recall')
    plt.legend(labels=['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))

    plt.subplot(8, 3, 10)
    plt.plot(range(1, 13), s['v12'])
    plt.plot(range(1, 25), s['v24'])
    plt.plot(range(1, 13), s['a12'])
    plt.plot(range(1, 25), s['a24'])
    plt.title('PFR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of First Recall')
    plt.legend(labels=['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_psr(s):
    plt.subplot(8, 3, 8)
    plt.plot(range(1, 13), s['f12'])
    plt.plot(range(1, 25), s['f24'])
    plt.plot(range(1, 13), s['s12'])
    plt.plot(range(1, 25), s['s24'])
    plt.title('PSR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Second Recall')
    plt.legend(labels=['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))

    plt.subplot(8, 3, 11)
    plt.plot(range(1, 13), s['v12'])
    plt.plot(range(1, 25), s['v24'])
    plt.plot(range(1, 13), s['a12'])
    plt.plot(range(1, 25), s['a24'])
    plt.title('PSR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Second Recall')
    plt.legend(labels=['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_ptr(s):
    plt.subplot(8, 3, 9)
    plt.plot(range(1, 13), s['f12'])
    plt.plot(range(1, 25), s['f24'])
    plt.plot(range(1, 13), s['s12'])
    plt.plot(range(1, 25), s['s24'])
    plt.title('PTR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Third Recall')
    plt.legend(labels=['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))

    plt.subplot(8, 3, 12)
    plt.plot(range(1, 13), s['v12'])
    plt.plot(range(1, 25), s['v24'])
    plt.plot(range(1, 13), s['a12'])
    plt.plot(range(1, 25), s['a24'])
    plt.title('PTR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Third Recall')
    plt.legend(labels=['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_pli_recency(s):
    plt.subplot(8, 3, 13)
    plt.plot(range(1, 7), s['f12'])
    plt.plot(range(1, 7), s['f24'])
    plt.plot(range(1, 7), s['s12'])
    plt.plot(range(1, 7), s['s24'])
    plt.title('PLI Recency')
    plt.xlabel('Number of Lists Back')
    plt.ylabel('Ratio of PLIs')
    plt.legend(labels=['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])

    plt.subplot(8, 3, 16)
    plt.plot(range(1, 7), s['v12'])
    plt.plot(range(1, 7), s['v24'])
    plt.plot(range(1, 7), s['a12'])
    plt.plot(range(1, 7), s['a24'])
    plt.title('PLI Recency')
    plt.xlabel('Number of Lists Back')
    plt.ylabel('Ratio of PLIs')
    plt.legend(labels=['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])


def plot_pli_early(s):
    plt.subplot(8, 3, 14)
    plt.bar([1, 2, 3, 4], [s['f12'], s['f24'], s['s12'], s['s24']], align='center', color='#D5D5D5')
    plt.xticks([1, 2, 3, 4], ['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])
    plt.title('PLI (Early)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. PLIs per List')

    plt.subplot(8, 3, 17)
    plt.bar([1, 2, 3, 4], [s['v12'], s['v24'], s['a12'], s['a24']], align='center', color='#D5D5D5')
    plt.xticks([1, 2, 3, 4], ['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])
    plt.title('PLI (Early)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. PLIs per List')


def plot_pli_late(s):
    plt.subplot(8, 3, 15)
    plt.bar([1, 2, 3, 4], [s['f12'], s['f24'], s['s12'], s['s24']], align='center', color='#D5D5D5')
    plt.xticks([1, 2, 3, 4], ['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])
    plt.title('PLI (Late)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. PLIs per List')

    plt.subplot(8, 3, 18)
    plt.bar([1, 2, 3, 4], [s['v12'], s['v24'], s['a12'], s['a24']], align='center', color='#D5D5D5')
    plt.xticks([1, 2, 3, 4], ['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])
    plt.title('PLI (Late)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. PLIs per List')


def plot_eli_early(s):
    plt.subplot(8, 3, 19)
    plt.bar([1, 2, 3, 4], [s['f12'], s['f24'], s['s12'], s['s24']], align='center', color='#D5D5D5')
    plt.xticks([1, 2, 3, 4], ['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])
    plt.title('ELI (Early)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. ELIs per List')

    plt.subplot(8, 3, 22)
    plt.bar([1, 2, 3, 4], [s['v12'], s['v24'], s['a12'], s['a24']], align='center', color='#D5D5D5')
    plt.xticks([1, 2, 3, 4], ['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])
    plt.title('ELI (Early)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. ELIs per List')


def plot_eli_late(s):
    plt.subplot(8, 3, 20)
    plt.bar([1, 2, 3, 4], [s['f12'], s['f24'], s['s12'], s['s24']], align='center', color='#D5D5D5')
    plt.xticks([1, 2, 3, 4], ['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])
    plt.title('ELI (Late)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. ELIs per List')

    plt.subplot(8, 3, 23)
    plt.bar([1, 2, 3, 4], [s['v12'], s['v24'], s['a12'], s['a24']], align='center', color='#D5D5D5')
    plt.xticks([1, 2, 3, 4], ['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])
    plt.title('ELI (Late)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. ELIs per List')


def plot_reps(s):
    plt.subplot(8, 3, 21)
    plt.bar([1, 2, 3, 4], [s['f12'], s['f24'], s['s12'], s['s24']], align='center', color='#D5D5D5')
    plt.xticks([1, 2, 3, 4], ['Fast/12', 'Fast/24', 'Slow/12', 'Slow/24'])
    plt.title('Repetitions')
    plt.xlabel('List Length')
    plt.ylabel('Avg. Repetitions per List')

    plt.subplot(8, 3, 24)
    plt.bar([1, 2, 3, 4], [s['v12'], s['v24'], s['a12'], s['a24']], align='center', color='#D5D5D5')
    plt.xticks([1, 2, 3, 4], ['Visual/12', 'Visual/24', 'Auditory/12', 'Auditory/24'])
    plt.title('Repetitions')
    plt.xlabel('List Length')
    plt.ylabel('Avg. Repetitions per List')


def plot_prec(s):
    pass
    '''
    plt.subplot()
    plt.bar([1, 2], [s[12], s[24]], align='center', color='#D5D5D5')
    plt.xticks([1, 2], ['12', '24'])
    plt.title('PRec')
    plt.xlabel('List Length')
    plt.ylabel('Probability of Recall')
    '''


def plot_irt(s):
    pass
