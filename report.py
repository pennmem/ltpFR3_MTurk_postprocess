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
    pli_early (Average prior list intrusions per list among the first three recalls)
    pli_late (Average prior list intrusions per list among recalls after the third)
    eli_early (Average extra list intrusions per list among the first three recalls)
    eli_late (Average extra list intrusions per list among recalls after the third)
    reps (Average number of repetitions per list)
    pli_recency (Ratio of prior list intrusions coming from each list back, up to 6 back)
    rec_per_trial (Array indicating how many words the participant correctly recalled on each trial)
    math_per_trial (Matrix indicating which math problems were answered correctly/incorrectly on each trial)
    
    And each entry contains a sub-dictionary with entries labelled 12, 18, and 24, which contain that stat for lists of
    length 12, 18, and 24, respectively. Note that length-18 lists are practice lists only.
    
    :param stats: A dictionary containing the behavioral stats calculated by run_stats.
    """
    stat_plotters = {'prec': plot_prec, 'spc': plot_spc, 'pfr': plot_pfr, 'psr': plot_psr, 'ptr': plot_ptr,
                      'crp_early': plot_crp_early, 'crp_late': plot_crp_late, 'pli_recency': plot_pli_recency,
                     'rec_per_trial': plot_rec_perlist, 'math_per_trial': plot_math_perlist}

    for subj in stats:
        pdf = PdfPages('/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/' + subj + '.pdf')
        plt.figure(figsize=(30, 30))
        if subj == 'all':
            plt.suptitle('All', fontsize=36)
            for key in stat_plotters:
                if key in stats['all']['mean']:
                    stat_plotters[key](stats['all']['mean'][key])
            plot_intrusions(stats['all']['mean']['plis'], stats['all']['mean']['elis'], stats['all']['mean']['reps'],
                            stats['all']['sem']['plis'], stats['all']['sem']['elis'], stats['all']['sem']['reps'])

        else:
            plt.suptitle(subj, fontsize=36)
            for key in stat_plotters:

                if key in stats[subj]:
                    stat_plotters[key](stats[subj][key])
                else:
                    print('ALERT! Missing stat %s for subject %s' % (key, subj))
            plot_intrusions(stats[subj]['plis'], stats[subj]['elis'], stats[subj]['reps'])



        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig()
        pdf.close()
        plt.close()


def plot_spc(s):
    plt.subplot(7, 3, 1)
    plt.plot(range(1, 13), s['s12'], 'k^--', markerfacecolor='white')
    plt.plot(range(1, 25), s['s24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['f12'], 'k^-')
    plt.plot(range(1, 25), s['f24'], 'ko-')
    plt.title('SPC')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Recall')
    plt.legend(labels=['Slow/12', 'Slow/24', 'Fast/12', 'Fast/24'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))

    plt.subplot(7, 3, 4)
    plt.plot(range(1, 13), s['a12'], 'k^--', markerfacecolor='white')
    plt.plot(range(1, 25), s['a24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['v12'], 'k^-')
    plt.plot(range(1, 25), s['v24'], 'ko-')
    plt.title('SPC')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Recall')
    plt.legend(labels=['Auditory/12', 'Auditory/24', 'Visual/12', 'Visual/24'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_crp_early(s):
    plt.subplot(7, 3, 2)
    plt.plot(range(-3, 4), s['s12'], 'k^--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['s24'], 'ko--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['f12'], 'k^-')
    plt.plot(range(-3, 4), s['f24'], 'ko-')
    plt.title('CRP (Early)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Slow/12', 'Slow/24', 'Fast/12', 'Fast/24'])
    plt.ylim(-.05, 1.05)

    plt.subplot(7, 3, 5)
    plt.plot(range(-3, 4), s['a12'], 'k^--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['a24'], 'ko--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['v12'], 'k^-')
    plt.plot(range(-3, 4), s['v24'], 'ko-')
    plt.title('CRP (Early)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Auditory/12', 'Auditory/24', 'Visual/12', 'Visual/24'])
    plt.ylim(-.05, 1.05)


def plot_crp_late(s):
    plt.subplot(7, 3, 3)
    plt.plot(range(-3, 4), s['s12'], 'k^--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['s24'], 'ko--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['f12'], 'k^-')
    plt.plot(range(-3, 4), s['f24'], 'ko-')
    plt.title('CRP (Late)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Slow/12', 'Slow/24', 'Fast/12', 'Fast/24'])
    plt.ylim(-.05, 1.05)

    plt.subplot(7, 3, 6)
    plt.plot(range(-3, 4), s['a12'], 'k^--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['a24'], 'ko--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['v12'], 'k^-')
    plt.plot(range(-3, 4), s['v24'], 'ko-')
    plt.title('CRP (Late)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Auditory/12', 'Auditory/24', 'Visual/12', 'Visual/24'])
    plt.ylim(-.05, 1.05)


def plot_pfr(s):
    plt.subplot(7, 3, 7)
    plt.plot(range(1, 13), s['s12'], 'k^--', markerfacecolor='white')
    plt.plot(range(1, 25), s['s24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['f12'], 'k^-')
    plt.plot(range(1, 25), s['f24'], 'ko-')
    plt.title('PFR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of First Recall')
    plt.legend(labels=['Slow/12', 'Slow/24', 'Fast/12', 'Fast/24'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))

    plt.subplot(7, 3, 10)
    plt.plot(range(1, 13), s['a12'], 'k^--', markerfacecolor='white')
    plt.plot(range(1, 25), s['a24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['v12'], 'k^-')
    plt.plot(range(1, 25), s['v24'], 'ko-')
    plt.title('PFR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of First Recall')
    plt.legend(labels=['Auditory/12', 'Auditory/24', 'Visual/12', 'Visual/24'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_psr(s):
    plt.subplot(7, 3, 8)
    plt.plot(range(1, 13), s['s12'], 'k^--', markerfacecolor='white')
    plt.plot(range(1, 25), s['s24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['f12'], 'k^-')
    plt.plot(range(1, 25), s['f24'], 'ko-')
    plt.title('PSR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Second Recall')
    plt.legend(labels=['Slow/12', 'Slow/24', 'Fast/12', 'Fast/24'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))

    plt.subplot(7, 3, 11)
    plt.plot(range(1, 13), s['a12'], 'k^--', markerfacecolor='white')
    plt.plot(range(1, 25), s['a24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['v12'], 'k^-')
    plt.plot(range(1, 25), s['v24'], 'ko-')
    plt.title('PSR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Second Recall')
    plt.legend(labels=['Auditory/12', 'Auditory/24', 'Visual/12', 'Visual/24'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_ptr(s):
    plt.subplot(7, 3, 9)
    plt.plot(range(1, 13), s['s12'], 'k^--', markerfacecolor='white')
    plt.plot(range(1, 25), s['s24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['f12'], 'k^-')
    plt.plot(range(1, 25), s['f24'], 'ko-')
    plt.title('PTR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Third Recall')
    plt.legend(labels=['Slow/12', 'Slow/24', 'Fast/12', 'Fast/24'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))

    plt.subplot(7, 3, 12)
    plt.plot(range(1, 13), s['a12'], 'k^--', markerfacecolor='white')
    plt.plot(range(1, 25), s['a24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['v12'], 'k^-')
    plt.plot(range(1, 25), s['v24'], 'ko-')
    plt.title('PTR')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Third Recall')
    plt.legend(labels=['Auditory/12', 'Auditory/24', 'Visual/12', 'Visual/24'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_pli_recency(s):
    plt.subplot(7, 3, 13)
    plt.plot(range(1, 7), s['s12'], 'k^--', markerfacecolor='white')
    plt.plot(range(1, 7), s['s24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 7), s['f12'], 'k^-')
    plt.plot(range(1, 7), s['f24'], 'ko-')
    plt.title('PLI Recency')
    plt.xlabel('Number of Lists Back')
    plt.ylabel('Ratio of PLIs')
    plt.legend(labels=['Slow/12', 'Slow/24', 'Fast/12', 'Fast/24'])
    plt.ylim(-.05, 1.05)

    plt.subplot(7, 3, 16)
    plt.plot(range(1, 7), s['a12'], 'k^--', markerfacecolor='white')
    plt.plot(range(1, 7), s['a24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 7), s['v12'], 'k^-')
    plt.plot(range(1, 7), s['v24'], 'ko-')
    plt.title('PLI Recency')
    plt.xlabel('Number of Lists Back')
    plt.ylabel('Ratio of PLIs')
    plt.legend(labels=['Auditory/12', 'Auditory/24', 'Visual/12', 'Visual/24'])
    plt.ylim(-.05, 1.05)


def plot_intrusions(plis, elis, reps, pli_err=None, eli_err=None, rep_err=None):
    plt.subplot(7, 3, 14)
    if None in (pli_err, eli_err, rep_err):
        plt.bar([1, 2, 3], [plis['all'], elis['all'], reps['all']], align='center', color='k', fill=False)
    else:
        pli_err = 1.96 * pli_err['all']
        eli_err = 1.96 * eli_err['all']
        rep_err = 1.96 * rep_err['all']
        plt.bar([1, 2, 3], [plis['all'], elis['all'], reps['all']], yerr=[pli_err, eli_err, rep_err], align='center',
                color='k', fill=False)
    plt.xticks([1, 2, 3], ['PLI', 'ELI', 'Rep'])
    plt.title('Intrusions')
    plt.ylabel('Intrusions Per List')


def plot_rec_perlist(s):
    plt.subplot(7, 3, 19)
    plt.plot(range(1, 19), s, 'ko-')
    plt.title('Recall Performance Check')
    plt.xlabel('Trial Number')
    plt.ylabel('Probability of Recall')
    plt.xlim(0, 19)
    plt.ylim(-.05, 1.05)


def plot_math_perlist(s):
    plt.subplot(7, 3, 21)
    plt.plot(range(1, 19), s, 'ko-')
    plt.title('Math Performance Check')
    plt.xlabel('Trial Number')
    plt.ylabel('Math Correct')
    plt.xlim(0, 19)
    plt.ylim(-.5, 20.5)


def plot_prec(s):
    pass


def plot_irt(s):
    pass
