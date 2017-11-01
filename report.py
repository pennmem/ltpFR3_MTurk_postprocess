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
    stat_plotters = {'spc': plot_spc, 'pfr': plot_pfr, 'psr': plot_psr, 'ptr': plot_ptr,
                     'crp_early': plot_crp_early, 'crp_late': plot_crp_late, 'pli_recency': plot_pli_recency,
                     'elis': plot_elis, 'plis': plot_plis, 'reps': plot_reps, 'rec_per_trial': plot_rec_perlist,
                     'math_per_trial': plot_math_perlist}

    for subj in stats:
        pdf = PdfPages('/data/eeg/scalp/ltp/ltpFR3_MTurk/reports/' + subj + '.pdf')
        #pdf = PdfPages('/Users/jessepazdera/Desktop/' + subj + '.pdf')
        plt.figure(figsize=(30, 30))
        if subj == 'all':
            plt.suptitle('All', fontsize=36)
            for key in stat_plotters:
                if key in stats['all']['mean']:
                    stat_plotters[key](stats['all']['mean'][key])
            plot_intrusions(stats['all']['mean']['plis'], stats['all']['mean']['elis'], stats['all']['mean']['reps'],
                            stats['all']['sem']['plis'], stats['all']['sem']['elis'], stats['all']['sem']['reps'])
            plot_elis(stats['all']['mean']['elis'], stats['all']['sem']['elis'])
            plot_plis(stats['all']['mean']['plis'], stats['all']['sem']['plis'])
            plot_reps(stats['all']['mean']['reps'], stats['all']['sem']['reps'])
        else:
            plt.suptitle(subj, fontsize=36)
            for key in stat_plotters:

                if key in stats[subj]:
                    stat_plotters[key](stats[subj][key])
                else:
                    print('ALERT! Missing stat %s for subject %s' % (key, subj))
            plot_intrusions(stats[subj]['plis'], stats[subj]['elis'], stats[subj]['reps'])
            plot_elis(stats[subj]['elis'])
            plot_plis(stats[subj]['plis'])
            plot_reps(stats[subj]['reps'])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig()
        pdf.close()
        plt.close()


def plot_spc(s):
    plt.subplot(7, 3, 1)
    plt.plot(range(1, 13), s['sv12'], 'ko-')
    plt.plot(range(1, 13), s['fv12'], 'k^-')
    plt.plot(range(1, 13), s['sa12'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['fa12'], 'k^--', markerfacecolor='white')
    plt.title('SPC (List Length 12)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Recall')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 12, 2), range(1, 12, 2))

    plt.subplot(7, 3, 4)
    plt.plot(range(1, 25), s['sv24'], 'ko-')
    plt.plot(range(1, 25), s['fv24'], 'k^-')
    plt.plot(range(1, 25), s['sa24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 25), s['fa24'], 'k^--', markerfacecolor='white')
    plt.title('SPC (List Length 24)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Recall')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_crp_early(s):
    plt.subplot(7, 3, 2)
    plt.plot(range(-3, 4), s['sv12'], 'ko-')
    plt.plot(range(-3, 4), s['fv12'], 'k^-')
    plt.plot(range(-3, 4), s['sa12'], 'ko--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['fa12'], 'k^--', markerfacecolor='white')
    plt.title('lag-CRP Early (List Length 12)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)

    plt.subplot(7, 3, 5)
    plt.plot(range(-3, 4), s['sv12'], 'ko-')
    plt.plot(range(-3, 4), s['fv12'], 'k^-')
    plt.plot(range(-3, 4), s['sa12'], 'ko--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['fa12'], 'k^--', markerfacecolor='white')
    plt.title('lag-CRP Early (List Length 24)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)


def plot_crp_late(s):
    plt.subplot(7, 3, 3)
    plt.plot(range(-3, 4), s['sv12'], 'ko-')
    plt.plot(range(-3, 4), s['fv12'], 'k^-')
    plt.plot(range(-3, 4), s['sa12'], 'ko--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['fa12'], 'k^--', markerfacecolor='white')
    plt.title('lag-CRP Late (List Length 12)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)

    plt.subplot(7, 3, 6)
    plt.plot(range(-3, 4), s['sv12'], 'ko-')
    plt.plot(range(-3, 4), s['fv12'], 'k^-')
    plt.plot(range(-3, 4), s['sa12'], 'ko--', markerfacecolor='white')
    plt.plot(range(-3, 4), s['fa12'], 'k^--', markerfacecolor='white')
    plt.title('lag-CRP Late (List Length 24)')
    plt.xlabel('Lag')
    plt.ylabel('Cond. Resp. Probability')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)


def plot_pfr(s):
    plt.subplot(7, 3, 7)
    plt.plot(range(1, 13), s['sv12'], 'ko-')
    plt.plot(range(1, 13), s['fv12'], 'k^-')
    plt.plot(range(1, 13), s['sa12'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['fa12'], 'k^--', markerfacecolor='white')
    plt.title('PFR (List Length 12)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of First Recall')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))

    plt.subplot(7, 3, 10)
    plt.plot(range(1, 25), s['sv24'], 'ko-')
    plt.plot(range(1, 25), s['fv24'], 'k^-')
    plt.plot(range(1, 25), s['sa24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 25), s['fa24'], 'k^--', markerfacecolor='white')
    plt.title('PFR (List Length 24)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of First Recall')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_psr(s):
    plt.subplot(7, 3, 8)
    plt.plot(range(1, 13), s['sv12'], 'ko-')
    plt.plot(range(1, 13), s['fv12'], 'k^-')
    plt.plot(range(1, 13), s['sa12'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['fa12'], 'k^--', markerfacecolor='white')
    plt.title('PSR (List Length 12)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Second Recall')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))

    plt.subplot(7, 3, 11)
    plt.plot(range(1, 25), s['sv24'], 'ko-')
    plt.plot(range(1, 25), s['fv24'], 'k^-')
    plt.plot(range(1, 25), s['sa24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 25), s['fa24'], 'k^--', markerfacecolor='white')
    plt.title('PSR (List Length 24)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Second Recall')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_ptr(s):
    plt.subplot(7, 3, 9)
    plt.plot(range(1, 13), s['sv12'], 'ko-')
    plt.plot(range(1, 13), s['fv12'], 'k^-')
    plt.plot(range(1, 13), s['sa12'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 13), s['fa12'], 'k^--', markerfacecolor='white')
    plt.title('PTR (List Length 12)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Third Recall')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))

    plt.subplot(7, 3, 12)
    plt.plot(range(1, 25), s['sv24'], 'ko-')
    plt.plot(range(1, 25), s['fv24'], 'k^-')
    plt.plot(range(1, 25), s['sa24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 25), s['fa24'], 'k^--', markerfacecolor='white')
    plt.title('PTR (List Length 24)')
    plt.xlabel('Serial Position')
    plt.ylabel('Probability of Third Recall')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, 1.05)
    plt.xticks(range(1, 25, 2), range(1, 25, 2))


def plot_pli_recency(s):
    plt.subplot(7, 3, 13)
    plt.plot(range(1, 7), s['sv12'], 'ko-')
    plt.plot(range(1, 7), s['fv12'], 'k^-')
    plt.plot(range(1, 7), s['sa12'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 7), s['fa12'], 'k^--', markerfacecolor='white')
    plt.title('PLI Recency')
    plt.xlabel('Number of Lists Back')
    plt.ylabel('Ratio of PLIs')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, .55)

    plt.subplot(7, 3, 16)
    plt.plot(range(1, 7), s['sv24'], 'ko-')
    plt.plot(range(1, 7), s['fv24'], 'k^-')
    plt.plot(range(1, 7), s['sa24'], 'ko--', markerfacecolor='white')
    plt.plot(range(1, 7), s['fa24'], 'k^--', markerfacecolor='white')
    plt.title('PLI Recency')
    plt.xlabel('Number of Lists Back')
    plt.ylabel('Ratio of PLIs')
    plt.legend(labels=['Slow/Visual', 'Fast/Visual', 'Slow/Auditory', 'Fast/Auditory'])
    plt.ylim(-.05, .55)


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


def plot_elis(s, err=None):
    plt.subplot(7, 3, 15)
    if err is None:
        plt.plot([1, 2], [s['sv12'], s['fv12']], 'ko-')
        plt.plot([1.02, 2.02], [s['sa12'], s['fa12']], 'ko--', markerfacecolor='white')
        plt.plot([3, 4], [s['sv24'], s['fv24']], 'ko-')
        plt.plot([3.02, 4.02], [s['sa24'], s['fa24']], 'ko--', markerfacecolor='white')
    else:
        plt.errorbar([1, 2], [s['sv12'], s['fv12']], yerr=[err['sv12'], err['fv12']], fmt='ko-')
        plt.errorbar([1.02, 2.02], [s['sa12'], s['fa12']], yerr=[err['sa12'], err['fa12']], fmt='ko--', markerfacecolor='white')
        plt.errorbar([3, 4], [s['sv24'], s['fv24']], yerr=[err['sv24'], err['fv24']], fmt='ko-')
        plt.errorbar([3.02, 4.02], [s['sa24'], s['fa24']], yerr=[err['sa24'], err['fa24']], fmt='ko--', markerfacecolor='white')
    plt.title('ELIs')
    plt.ylabel('ELIs Per List')
    plt.xticks([1, 2, 3, 4], ('12/Slow', '12/Fast', '24/Slow', '24/Fast'))
    plt.legend(labels=['Visual', 'Auditory'])


def plot_plis(s, err=None):
    plt.subplot(7, 3, 17)
    if err is None:
        plt.plot([1, 2], [s['sv12'], s['fv12']], 'ko-')
        plt.plot([1.02, 2.02], [s['sa12'], s['fa12']], 'ko--', markerfacecolor='white')
        plt.plot([3, 4], [s['sv24'], s['fv24']], 'ko-')
        plt.plot([3.02, 4.02], [s['sa24'], s['fa24']], 'ko--', markerfacecolor='white')
    else:
        plt.errorbar([1, 2], [s['sv12'], s['fv12']], yerr=[err['sv12'], err['fv12']], fmt='ko-')
        plt.errorbar([1.02, 2.02], [s['sa12'], s['fa12']], yerr=[err['sa12'], err['fa12']], fmt='ko--', markerfacecolor='white')
        plt.errorbar([3, 4], [s['sv24'], s['fv24']], yerr=[err['sv24'], err['fv24']], fmt='ko-')
        plt.errorbar([3.02, 4.02], [s['sa24'], s['fa24']], yerr=[err['sa24'], err['fa24']], fmt='ko--', markerfacecolor='white')
    plt.title('PLIs')
    plt.ylabel('PLIs Per List')
    plt.xticks([1, 2, 3, 4], ('12/Slow', '12/Fast', '24/Slow', '24/Fast'))
    plt.legend(labels=['Visual', 'Auditory'])


def plot_reps(s, err=None):
    plt.subplot(7, 3, 18)
    if err is None:
        plt.plot([1, 2], [s['sv12'], s['fv12']], 'ko-')
        plt.plot([1.02, 2.02], [s['sa12'], s['fa12']], 'ko--', markerfacecolor='white')
        plt.plot([3, 4], [s['sv24'], s['fv24']], 'ko-')
        plt.plot([3.02, 4.02], [s['sa24'], s['fa24']], 'ko--', markerfacecolor='white')
    else:
        plt.errorbar([1, 2], [s['sv12'], s['fv12']], yerr=[err['sv12'], err['fv12']], fmt='ko-')
        plt.errorbar([1.02, 2.02], [s['sa12'], s['fa12']], yerr=[err['sa12'], err['fa12']], fmt='ko--', markerfacecolor='white')
        plt.errorbar([3, 4], [s['sv24'], s['fv24']], yerr=[err['sv24'], err['fv24']], fmt='ko-')
        plt.errorbar([3.02, 4.02], [s['sa24'], s['fa24']], yerr=[err['sa24'], err['fa24']], fmt='ko--', markerfacecolor='white')
    plt.title('Repetitions')
    plt.ylabel('Reps Per List')
    plt.xticks([1, 2, 3, 4], ('12/Slow', '12/Fast', '24/Slow', '24/Fast'))
    plt.legend(labels=['Visual', 'Auditory'])


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


if __name__ == "__main__":
    import json
    with open('/Users/jessepazdera/Desktop/ltpFR3_stats.json') as f:
        stats = json.load(f)
    ltpFR3_report(stats)
