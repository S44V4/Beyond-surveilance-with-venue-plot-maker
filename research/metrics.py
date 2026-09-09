"""Pure metrics and paired sequence-cluster uncertainty; no model dependencies."""
import numpy as np

COUNT_METRICS=['zone_count_mae','zone_count_rmse','total_count_mae','total_count_rmse']
def confusion(truth,pred,classes=4):
    truth=np.asarray(truth,dtype=int).reshape(-1);pred=np.asarray(pred,dtype=int).reshape(-1)
    return np.bincount(truth*classes+pred,minlength=classes**2).reshape(classes,classes)

def pressure_scores(matrix):
    matrix=np.asarray(matrix,dtype=float);tp=matrix.diagonal()
    recall=np.divide(tp,matrix.sum(1),out=np.zeros(4),where=matrix.sum(1)>0)
    precision=np.divide(tp,matrix.sum(0),out=np.zeros(4),where=matrix.sum(0)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(4),where=precision+recall>0)
    return {'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),'accuracy':float(tp.sum()/max(matrix.sum(),1)),'per_class_precision':precision.tolist(),'per_class_recall':recall.tolist(),'per_class_f1':f1.tolist(),'support':matrix.sum(1).astype(int).tolist()}

def summarize(truth,pred,thresholds):
    truth=np.asarray(truth);pred=np.asarray(pred)
    if truth.shape!=pred.shape or truth.shape[-1]!=2:raise ValueError('Expect aligned [window,zone,count/pressure] arrays')
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():raise ValueError('Non-finite predictions')
    delta=pred-truth;count=delta[...,0];total=count.sum(-1);pressure=delta[...,1]
    cm=confusion(np.digitize(truth[...,1],thresholds),np.digitize(pred[...,1],thresholds))
    result={'zone_count_mae':float(np.abs(count).mean()),'zone_count_mse':float((count**2).mean()),'zone_count_rmse':float(np.sqrt((count**2).mean())),'total_count_mae':float(np.abs(total).mean()),'total_count_mse':float((total**2).mean()),'total_count_rmse':float(np.sqrt((total**2).mean())),'pressure_mae':float(np.abs(pressure).mean()),'pressure_mse':float((pressure**2).mean()),'pressure_rmse':float(np.sqrt((pressure**2).mean())),'pressure_confusion':cm.tolist(),'windows':len(truth),**{f'pressure_{k}':v for k,v in pressure_scores(cm).items()}}
    return result

def cluster_interval(learned,baseline,rmse=False,seed=20260907,replicates=10000):
    """Inputs [training seed, sequence]; MSE inputs when rmse=True.

    Average seed-specific scores, never predictions; resample paired sequences
    with the same indices across models and seeds. All correlated windows in a
    sequence stay together. This is conditional uncertainty for this split.
    """
    learned=np.asarray(learned,dtype=float);baseline=np.asarray(baseline,dtype=float)
    if learned.shape!=baseline.shape or learned.ndim!=2 or learned.shape[1]<2:raise ValueError('Need paired seed-by-sequence values and at least two sequences')
    rng=np.random.default_rng(seed);indices=rng.integers(0,learned.shape[1],size=(replicates,learned.shape[1]))
    a=learned[:,indices].mean(-1);b=baseline[:,indices].mean(-1)
    point_a=learned.mean(1);point_b=baseline.mean(1)
    if rmse:a=np.sqrt(a);b=np.sqrt(b);point_a=np.sqrt(point_a);point_b=np.sqrt(point_b)
    delta=(a-b).mean(0)
    return {'learned':float(point_a.mean()),'persistence':float(point_b.mean()),'delta':float((point_a-point_b).mean()),'delta_ci95':np.quantile(delta,[.025,.975]).tolist(),'simultaneous_upper_one_sided95':float(np.quantile(delta,1-.05/24)),'learned_ci95':np.quantile(a.mean(0),[.025,.975]).tolist(),'persistence_ci95':np.quantile(b.mean(0),[.025,.975]).tolist(),'independent_units':learned.shape[1],'training_seeds':learned.shape[0],'replicates':replicates}
