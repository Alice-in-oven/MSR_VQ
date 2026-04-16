from collections import OrderedDict

def eval_recall(pre, gt, ranks=[[1, 1], [1, 10], [10, 10], [1, 100], [100, 100]]):
    recalls = []
    for rank in ranks:
        found = 0
        for i in range(0, pre.shape[0]):
            result_set_set = set(pre[i][0:rank[1]])
            truth_set_set = set(gt[i][0:rank[0]])
            found += len(result_set_set.intersection(truth_set_set))
        recall = found / (pre.shape[0] * rank[0])
        recalls.append(('recall {}@{}'.format(rank[0], rank[1]), recall))
    recalls = OrderedDict(recalls)
    return recalls