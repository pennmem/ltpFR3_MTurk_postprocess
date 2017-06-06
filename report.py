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
    nback_pli_rate (Ratio of prior list intrusions coming from each list back, up to 6 back)
    
    And each entry contains a sub-dictionary with entries labelled 12, 18, and 24, which contain that stat for lists of
    length 12, 18, and 24, respectively. Note that length-18 lists are practice lists only.
    
    :param stats: A dictionary containing the behavioral stats calculated by run_stats.
    """
    stat_plotters = {'prec': plot_prec, 'spc': plot_spc, 'pfr': plot_pfr, 'psr': plot_psr, 'ptr': plot_ptr,
                      'crp_early': plot_crp_early, 'crp_late': plot_crp_late, 'irt': plot_irt,
                      'pli_early': plot_pli_early, 'pli_late': plot_pli_late, 'eli_early': plot_eli_early,
                      'eli_late': plot_eli_late, 'reps': plot_reps, 'nback_pli_rate': plot_nback_pli_rate}

    stat_order = ['prec', 'spc', 'pfr', 'psr', 'ptr', 'crp_early', 'crp_late', 'irt', 'pli_early', 'pli_late', 'nback_pli_rate', 'eli_early', 'eli_late', 'reps']

    for subj in stats:
        pdf = PdfPages('/Users/jessepazdera/Desktop/' + subj + '.pdf')
        plt.figure(figsize=(40, 30))
        plt.suptitle(subj, fontsize=36)
        for key in stat_order:
            if key in stats[subj]:
                stat_plotters[key](stats[subj][key])
            else:
                print 'ALERT! Missing stat %s for subject %s' % (key, subj)
        pdf.savefig()
        pdf.close()
        plt.close()


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


def plot_spc(s):
    plt.subplot(4, 3, 1)
    plt.plot(range(1, 13), s[12], '-ko')
    plt.plot(range(1, 25), s[24], '--kD')
    plt.title('SPC')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Recall')
    plt.legend(labels=['Length 12', 'Length 24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_pfr(s):
    plt.subplot(4, 3, 4)
    plt.plot(range(1, 13), s[12], '-ko')
    plt.plot(range(1, 25), s[24], '--kD')
    plt.title('PFR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of First Recall')
    plt.legend(labels=['Length 12', 'Length 24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_psr(s):
    plt.subplot(4, 3, 5)
    plt.plot(range(1, 13), s[12], '-ko')
    plt.plot(range(1, 25), s[24], '--kD')
    plt.title('PSR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Second Recall')
    plt.legend(labels=['Length 12', 'Length 24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_ptr(s):
    plt.subplot(4, 3, 6)
    plt.plot(range(1, 13), s[12], '-ko')
    plt.plot(range(1, 25), s[24], '--kD')
    plt.title('PTR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Third Recall')
    plt.legend(labels=['Length 12', 'Length 24'])
    plt.xlim(1, 24)
    plt.ylim(0, 1)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_crp_early(s):
    plt.subplot(4, 3, 2)
    plt.plot(range(-3, 4), s[12], '-ko')
    plt.plot(range(-3, 4), s[24], '--kD')
    plt.title('CRP (Early)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Length 12', 'Length 24'])
    plt.ylim(0, 1)


def plot_crp_late(s):
    plt.subplot(4, 3, 3)
    plt.plot(range(-3, 4), s[12], '-ko')
    plt.plot(range(-3, 4), s[24], '--kD')
    plt.title('CRP (Late)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Length 12', 'Length 24'])
    plt.ylim(0, 1)


def plot_irt(s):
    pass


def plot_pli_early(s):
    plt.subplot(4, 3, 7)
    plt.bar([1, 2], [s[12], s[24]], align='center', color='#D5D5D5')
    plt.xticks([1, 2], ['12', '24'])
    plt.title('PLI (Early)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. PLIs per List')


def plot_pli_late(s):
    plt.subplot(4, 3, 8)
    plt.bar([1, 2], [s[12], s[24]], align='center', color='#D5D5D5')
    plt.xticks([1, 2], ['12', '24'])
    plt.title('PLI (Late)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. PLIs per List')


def plot_eli_early(s):
    plt.subplot(4, 3, 10)
    plt.bar([1, 2], [s[12], s[24]], align='center', color='#D5D5D5')
    plt.xticks([1, 2], ['12', '24'])
    plt.title('ELI (Early)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. ELIs per List')


def plot_eli_late(s):
    plt.subplot(4, 3, 11)
    plt.bar([1, 2], [s[12], s[24]], align='center', color='#D5D5D5')
    plt.xticks([1, 2], ['12', '24'])
    plt.title('ELI (Late)')
    plt.xlabel('List Length')
    plt.ylabel('Avg. ELIs per List')


def plot_reps(s):
    plt.subplot(4, 3, 12)
    plt.bar([1, 2], [s[12], s[24]], align='center', color='#D5D5D5')
    plt.xticks([1, 2], ['12', '24'])
    plt.title('Reps')
    plt.xlabel('List Length')
    plt.ylabel('Avg. Repetitions per List')


def plot_nback_pli_rate(s):
    plt.subplot(4, 3, 9)
    plt.plot(range(1, 7), s[12], '-ko')
    plt.plot(range(1, 7), s[24], '--kD')
    plt.title('N-Back PLI Rate')
    plt.xlabel('Number of Lists Back')
    plt.ylabel('Ratio of PLIs')
    plt.legend(labels=['Length 12', 'Length 24'])
