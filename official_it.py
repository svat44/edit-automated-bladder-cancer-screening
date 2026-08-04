# INTENSITY THRESHOLDING BASELINE - extracted from 2_Intensity_Approach_v2.ipynb
# EDIT AI Bladder Cancer Detection | intensity (NC-ratio) baseline vs deep-learning model


#main segmentation code taken from the project packet and implemented otsu variations + classification metrics
import time
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('image', cmap='gray')
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import f1_score, classification_report
from intensity_utils import predict_nucleus_cytoplasm_masks,calc_nc_ratio_from_image,get_nucleus_cytoplasm_masks,calculate_true_NC_ratios,plot_pred_vs_gt,plot_segmentation_breakdown,plot_confusion_analysis,get_nc_ratio_from_mask, mean_absolute_error
from skimage.morphology import opening, closing, disk
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

urothelial_cells=pd.read_pickle("urothelial_cell_toy_data.pkl")
images=np.transpose(urothelial_cells["X"].numpy()*255,(0,2,3,1)).astype(np.uint8)
labels=urothelial_cells["y"]
no_nucleus=(labels==2).sum(axis=(1,2))==0
images=images[~no_nucleus]
labels=labels[~no_nucleus]
images_gray=np.array([rgb2gray(each_image/255.) for each_image in images])

print("loaded:", len(labels))
no_nucleus = (labels == 2).sum(axis=(1,2)) == 0
print("dropped (no nucleus):", no_nucleus.sum())
images = images[~no_nucleus]; labels = labels[~no_nucleus]
print("after filter:", len(labels))
time.sleep(1)

X_train_val,X_test,Y_train_val,Y_test=train_test_split(images_gray,labels,test_size=0.2,stratify=labels.mean((1,2))>=np.median(labels.mean((1,2))),random_state=42)
X_train,X_val,Y_train,Y_val=train_test_split(X_train_val,Y_train_val,test_size=0.2,stratify=Y_train_val.mean((1,2))>=np.median(Y_train_val.mean((1,2))),random_state=42)

idx=2
plt.subplot(1,2,1)
plt.imshow(X_train[idx])
plt.subplot(1,2,2)
plt.imshow(Y_train[idx])
plt.show()

plt.subplot(121)
plt.imshow(Y_train[idx]==1,cmap="Greens")
plt.subplot(122)
plt.imshow(Y_train[idx]==2,cmap="Blues")
plt.show()


nc_ratio_train_true,nc_ratio_val_true,nc_ratio_test_true=calculate_true_NC_ratios(Y_train,Y_val,Y_test)


nucleus_threshold=0.4
cytoplasm_threshold=0.8

nucleus_mask,cytoplasm_mask,y_pred=predict_nucleus_cytoplasm_masks(X_train[idx],nucleus_threshold,cytoplasm_threshold,postprocess=False,return_pred=True)

nucleus_gt,cytoplasm_gt=get_nucleus_cytoplasm_masks(Y_train[idx])

plot_pred_vs_gt(cytoplasm_mask, nucleus_mask,
                cytoplasm_gt, nucleus_gt)
plt.show()


nucleus_threshold=0.2
cytoplasm_threshold=0.9

nucleus_mask,cytoplasm_mask,y_pred=predict_nucleus_cytoplasm_masks(X_train[idx],nucleus_threshold,cytoplasm_threshold,postprocess=False,return_pred=True)

nucleus_gt,cytoplasm_gt=get_nucleus_cytoplasm_masks(Y_train[idx])

plot_pred_vs_gt(cytoplasm_mask, nucleus_mask,
                cytoplasm_gt, nucleus_gt)
plt.show()

plot_segmentation_breakdown(cytoplasm_mask, nucleus_mask,
                            cytoplasm_gt, nucleus_gt)
plt.show()

plot_confusion_analysis(Y_train[idx], y_pred)
plt.show()

y_true=Y_train[idx].flatten()
print(classification_report(y_true,y_pred.flatten(),target_names=["Background","Cytoplasm","Nucleus"]))

y_true=Y_train[idx].flatten()
nucleus_thresholds=np.arange(0,1.1,0.1)
cytoplasm_thresholds=np.arange(0,1.1,0.1)
results_list=[]
for nucleus_threshold in nucleus_thresholds:
    for cytoplasm_threshold in cytoplasm_thresholds:
        if nucleus_threshold<cytoplasm_threshold:
            nucleus_mask,cytoplasm_mask,y_pred=predict_nucleus_cytoplasm_masks(X_train[idx],nucleus_threshold,cytoplasm_threshold,postprocess=False,return_pred=True)
            results_list.append([round(nucleus_threshold,1),round(cytoplasm_threshold,1),round(f1_score(y_true,y_pred.flatten(),average="macro"),2)*100])
results_df=pd.DataFrame(results_list,columns=["nucleus_threshold","cytoplasm_threshold","F1 Score"])
print(results_df.sort_values("F1 Score",ascending=False).head(15))


top_nucleus_threshold,top_cytoplasm_threshold=results_df.sort_values("F1 Score",ascending=False).iloc[0][["nucleus_threshold","cytoplasm_threshold"]]

idx_new=99
nucleus_mask,cytoplasm_mask,y_pred=predict_nucleus_cytoplasm_masks(X_train[idx_new],top_nucleus_threshold,top_cytoplasm_threshold,postprocess=False,return_pred=True)

nucleus_gt,cytoplasm_gt=get_nucleus_cytoplasm_masks(Y_train[idx_new])
plt.figure()
plt.imshow(X_train[idx_new])

plot_pred_vs_gt(cytoplasm_mask, nucleus_mask,
                cytoplasm_gt, nucleus_gt)
plt.show()

y_true=Y_train[idx_new].flatten()
print(classification_report(y_true,y_pred.flatten(),target_names=["Background","Cytoplasm","Nucleus"]))


nucleus_threshold=0.4
cytoplasm_threshold=0.8

nucleus_mask,cytoplasm_mask,y_pred=predict_nucleus_cytoplasm_masks(X_train[idx],nucleus_threshold,cytoplasm_threshold,postprocess=True,return_pred=True)

nucleus_gt,cytoplasm_gt=get_nucleus_cytoplasm_masks(Y_train[idx])

plot_pred_vs_gt(cytoplasm_mask, nucleus_mask,
                cytoplasm_gt, nucleus_gt)
plt.show()


# ## Calculate NC Ratio

print(get_nc_ratio_from_mask(y_true),get_nc_ratio_from_mask(y_pred),calc_nc_ratio_from_image(X_train[idx],nucleus_threshold,cytoplasm_threshold,postprocess=True))



thresholds=(nucleus_threshold,cytoplasm_threshold)
nc_ratio_test_pred=np.array([calc_nc_ratio_from_image(each_image,nucleus_threshold,cytoplasm_threshold) for each_image in X_test])
stat=float(spearmanr(nc_ratio_test_pred,nc_ratio_test_true,nan_policy="omit")[0])
print(stat,mean_absolute_error(nc_ratio_test_true,nc_ratio_test_pred))

plt.scatter(nc_ratio_test_pred,nc_ratio_test_true)
plt.plot([0,1],[0,1],ls="--")
plt.xlabel("Predicted NC")
plt.ylabel("True NC")
sns.despine()
plt.show()


#Optimized hresholds with Optuna and skopt
try:
    import optuna # sorry for weird import lol
    optuna_available = True
except ImportError:
    optuna = None
    optuna_available = False
    print("no optuna found; skip optimization")

if optuna_available:
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    import warnings
    warnings.filterwarnings("ignore")

    def callback(study, trial, metric):
        best = study.best_trial
        val = -best.value if metric == 'spearman' else best.value
        print(f"Iter {len(study.trials)} | Best: {val:.4f} | x={sorted(best.params.values())}")

    metric_dict = dict(
        spearman=lambda x, y: spearmanr(x, y, nan_policy="omit")[0],
        mae=lambda x, y: -mean_absolute_error(x, y)
    )

    def loss_func(thresholds, metric="spearman"):
        # thresholds = sorted(thresholds)

        nc_ratio_train_pred = np.vectorize(
            lambda x: calc_nc_ratio_from_image(x, *thresholds, postprocess=False),
            signature='(w,h)->()'
        )(X_train)

        if np.sum(~np.isnan(nc_ratio_train_pred)) < 3:
            return 1

        stat = metric_dict[metric](nc_ratio_train_true, nc_ratio_train_pred)

        if np.isnan(stat) and metric == "spearman":
            return 0
        elif np.isnan(stat) and metric == "mae":
            return 1

        return -stat

    def make_objective(metric):
        def objective(trial):
            t1 = trial.suggest_float("threshold1", 0, 0.99) # , step=0.0001
            t2 = trial.suggest_float("threshold2", t1, 0.99) # , step=0.0001
            return loss_func([t1, t2], metric=metric)
        return objective

    metric = "mae"

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(direction="minimize",sampler=sampler)

    study.optimize(
        make_objective(metric),
        n_trials=150,
        n_jobs=4,
        callbacks=[lambda study, trial: callback(study, trial, metric)]
    )

    class Result:
        pass

    res = Result()
    res.x = list(study.best_trial.params.values())
    res.fun = study.best_value
    res.x_iters = [list(t.params.values()) for t in study.trials]
    res.func_vals = np.array([t.value for t in study.trials])

    print("\nFinal result:")
    print("Best x:", sorted(res.x))
    print("Best value:", (-res.fun if metric == "spearman" else res.fun))

    final_thresholds=sorted(res.x_iters[res.func_vals.argmin()])
    pd.to_pickle(final_thresholds,"final_thresholds.pkl")

    nc_ratio_test_pred=np.vectorize(lambda x: calc_nc_ratio_from_image(x,*final_thresholds,postprocess=True),signature='(w,h)->()')(X_test)
    print(spearmanr(nc_ratio_test_pred,nc_ratio_test_true,nan_policy="omit")[0],np.abs(nc_ratio_test_pred-nc_ratio_test_true).mean())

    plt.scatter(nc_ratio_test_pred,nc_ratio_test_true)
    plt.plot([0,1],[0,1],ls="--")
    plt.xlabel("Predicted NC")
    plt.ylabel("True NC")
    sns.despine()
    plt.show()
else:
    print("Skipping optuna optimization because optuna is not available.")

# separate optimization with skopt

try:
    from skopt import gp_minimize
    from skopt.space import Real
    skopt_available = True
except ImportError:
    gp_minimize = None
    Real = None
    skopt_available = False
    print("skopt is not installed; skopt optimization will be skipped.")

if skopt_available:
    def callback_skopt(res,metric):
        print(f"Iter {len(res.x_iters)} | Best: {(-res.fun if metric=='spearman' else res.fun):.4f} | x={sorted(res.x)}")

    metric_dict_skopt = dict(spearman=lambda x,y: spearmanr(x,y,nan_policy="omit")[0],
                    mae=lambda x,y: -mean_absolute_error(x,y))

    def loss_func_skopt(thresholds,metric="spearman"):
        thresholds=sorted(thresholds)
        nc_ratio_train_pred=np.vectorize(lambda x: calc_nc_ratio_from_image(x,*thresholds,postprocess=False),signature='(w,h)->()')(X_train)
        if np.sum(~np.isnan(nc_ratio_train_pred))<3: return 1
        stat=metric_dict_skopt[metric](nc_ratio_train_true,nc_ratio_train_pred)
        if np.isnan(stat) and metric=="spearman": return 0
        elif np.isnan(stat) and metric=="mae": return 1
        return -stat

    space = [
        Real(0, 0.99, name='threshold1'),
        Real(0, 0.99, name='threshold2')
    ]

    metric="mae"
    # Perform the optimization
    res = gp_minimize(lambda x: loss_func_skopt(x,metric=metric), space, x0=(0.2,0.8), n_calls=50, acq_optimizer="lbfgs", acq_func="EI", noise=1e-1, n_restarts_optimizer=15, random_state=0, n_jobs=4, callback=[lambda res: callback_skopt(res,metric)], verbose=False)
else:
    print("Skipping skopt optimization because skopt is not available.")


import itertools
from skimage.filters import threshold_multiotsu
from sklearn.model_selection import StratifiedKFold

SPEC_CLASSES = ["neg", "aty", "sus", "pos"]          # ordinal 0 < 1 < 2 < 3


def nc_ratio_multiotsu(img):
    """Predicted NC ratio via per-image 3-level Otsu. img = HxWx3 uint8/float."""
    g = rgb2gray(np.asarray(img) / 255.)             # 0 = dark (nucleus) .. 1 = bright (bg)
    try:
        t = threshold_multiotsu(g, classes=3)      
    except ValueError:                               
        return np.nan
    reg = np.digitize(g, bins=t)                     # 0 darkest (nucleus), 1 cytoplasm, 2 background
    nuc = int((reg == 0).sum())
    cyto = int((reg == 1).sum())
    return nuc / (nuc + cyto) if (nuc + cyto) > 0 else np.nan


spec = pd.read_pickle("specimens_toy_data.pkl")

nc_pred, nc_given, y = [], [], []
for cls_idx, cat in enumerate(SPEC_CLASSES):
    md = spec[cat]["metadata"].reset_index(drop=True)
    for j, img in enumerate(spec[cat]["imgs"]):
        nc_pred.append(nc_ratio_multiotsu(img))
        nc_given.append(md.loc[j, "nc_ratio"])       # provided NC = oracle
        y.append(cls_idx)

nc_pred  = np.array(nc_pred, dtype=float)
nc_given = np.array(nc_given, dtype=float)
y        = np.array(y)

print("\npredicted (multi-Otsu) vs given NC, per-class mean:")
for i, cat in enumerate(SPEC_CLASSES):
    m = y == i
    print(f"  {cat}: pred {np.nanmean(nc_pred[m]):.3f}  given {nc_given[m].mean():.3f}  "
          f"(NaN preds: {int(np.isnan(nc_pred[m]).sum())})")
print(f"Spearman(pred, given) = "
      f"{spearmanr(nc_pred, nc_given, nan_policy='omit')[0]:.3f}")

def bin_by_cuts(s, cuts):
    t1, t2, t3 = cuts
    out = np.zeros(len(s), dtype=int)
    out[s >= t1] = 1; out[s >= t2] = 2; out[s >= t3] = 3
    return out


def fit_cuts(s, yt, grid=None):
    if grid is None:
        grid = np.round(np.arange(0.05, 0.70, 0.02), 3)
    best, bq = None, -2.0
    for c in itertools.combinations(grid, 3):
        q = cohen_kappa_score(yt, bin_by_cuts(s, c), weights="quadratic")
        if q > bq:
            bq, best = q, c
    return best


def cv_evaluate(scores, y, tag):
    scores = np.nan_to_num(scores.copy(), nan=0.0)   # undefined NC -> lowest bin
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(scores, y):
        pred[te] = bin_by_cuts(scores[te], fit_cuts(scores[tr], y[tr]))
    qwk = cohen_kappa_score(y, pred, weights="quadratic")
    print(f"\n########## {tag} ##########")
    print(f"Quadratic weighted kappa: {qwk:.3f}")
    print("Confusion (rows=true, cols=pred), order", SPEC_CLASSES)
    print(confusion_matrix(y, pred, labels=[0, 1, 2, 3]))
    print(classification_report(y, pred, labels=[0, 1, 2, 3],
                                target_names=SPEC_CLASSES, zero_division=0))


cv_evaluate(nc_given, y, "ORACLE: provided NC ratio (ceiling)")
cv_evaluate(nc_pred,  y, "PREDICTED: per-image multi-Otsu (end-to-end baseline)")