"""
%IRT Inter-response time.
%
% [irts, uniq_index] = irt(times_matrix, index, from_mask, to_mask)
%
% INPUTS:
% times_matrix: a matrix whose elements are times of recalled
%  items, relative to the start of a recall period during
%  a trial. The rows of this matrix should
%  represent times of recalls by a single subject
%  on a single trial.
%
%  index: a column vector which indexes the rows of
%  times_matrix with a subject number (or other
%  identifier).
%
%  A typical way of indexing the times matrix is to
%  group by both subject and by number of outputs, or
%  number of correct recalls. If this is the
%  behavior you would like, use:
%
%  num_ops = number_of_outputs(mask)
%  index = make_index(subjects, num_ops)
%
%  where mask is the mask passed to this function,
%  and subjects is a column vector containing subject
%  labels for times_matrix.
%
%  from_mask: if given, a logical matrix of the same shape as
%  times_matrix, which is false at positions (i, j)
%  where the value in times_matrix(i,j) should not be
%  used as the 'from' point in an IRT calculation
%  (e.g., it is the time of an intrusion event). If
%  no mask is given, a blank mask is used.
%
%  to_mask: if given, a logical matrix of the same shape as
%  times_matrix, which is false at positions (i, j)
%  where the value in times_matrix(i,j) should not be
%  used as the 'to' point in an IRT calculation
%  (e.g., it is the time of an intrusion event). If
%  no mask is given, the from_mask is used.
%
% OUTPUTS:
% irts: a matrix whose rows contain mean inter-response times
%  for each of the unique values in index
%
% uniq_index: a column vector which indexes the irts matrix
%  with the unique values of the input index. For
%  example, if trials where subject 1 had three
%  outputs corresponds to index 5, then the mean
%  irts for that subject on those trials are in:
%  irts(uniq_index == 5, :smile:
%
% NOTES:
%  If you are using an index that groups by more than one condition
%  (as is typical), be sure to collect the second output
%  (uniq_index) if you want to apply further transformations (e.g.,
%  mean across subject or across number of outputs).
%
% EXAMPLES:
% mask = make_clean_recalls_mask2d(recalls_matrix);
% num_ops = number_of_outputs(mask);
% [index, index_values_cell] = make_index(subjects, num_ops);
% [irts, uniq_index] = irt(times_matrix, index, list_length, mask);
%
% % take the mean across subjects within number of outputs
% % by using the second column of index_values_cell as an index
% op_index = [index_values_cell{:, 2}]';
% [mean_irts_by_op, op_vals] = apply_by_index(@nanmean, op_index, ...
% 1, {irts}, 1);
% plot(1:size(mean_irts_by_op, 2), mean_irts_by_op)
"""
import numpy as np, h5py
import scipy.io as sio
from copy import deepcopy

def irt(times=None, subjects=None, listLength=None, lag_num=None):
    irt = deepcopy(times)
    for num, item in enumerate(times):
        for index,recall in enumerate(item):
            if recall == 0:
                irt[num][index-1] = 0
            elif index == len(item) - 1:
                irt[num][index - 1] = times[num][index] - times[num][index - 1]
                irt[num][index] = 0
            elif index != 0:
                irt[num][index-1] = times[num][index] - times[num][index - 1]

    return irt


"""times = []
subj = []

for n in range(63, 65):
    print (n)

    if n < 100:
        subj_num = '0' + str(n)
    else:
        subj_num = str(n)
    try:
        files = sio.loadmat('/Users/janglim/rhino/data/eeg/scalp/ltp/ltpFR/behavioral/data/stat_data_LTP' + subj_num + '.mat', squeeze_me = True, struct_as_record=False)
        if set(range(8,15)).issubset(files['data'].session):
            for item in files['data'].times:
                times.append(item.astype('int').tolist())
            for item in files['data'].subject:
                subj.append(item.astype('int').tolist())
    except FileNotFoundError:
        continue


print(times)
print(irt(times, listLength=16))"""