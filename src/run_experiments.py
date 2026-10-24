"""Reproducible hyperparameter optimization project (dataset is packaged in sklearn)."""
from pathlib import Path
import json, platform, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, ParameterSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results'; FIG = ROOT / 'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEED=42; CV_FOLDS=5; N_TRIALS=30

raw=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(raw.data,raw.target,test_size=.2,stratify=raw.target,random_state=SEED)
cv=StratifiedKFold(n_splits=CV_FOLDS,shuffle=True,random_state=SEED)
space={'n_estimators':[20,40,60,80,100], 'max_depth':[None,3,5,10,15,20,30], 'min_samples_split':[2,4,6,10,15,20], 'min_samples_leaf':[1,2,3,5,7,10]}

def score_cv(params, seed=SEED):
    model=RandomForestClassifier(**params,random_state=seed,n_jobs=1)
    t=time.perf_counter()
    scores=cross_val_score(model,X_train,y_train,scoring='f1_macro',cv=cv,n_jobs=1)
    return float(scores.mean()),float(scores.std()),time.perf_counter()-t

baseline_params={}
base_mean,base_std,base_search_time=score_cv(baseline_params)
trials=[];best=-1.; best_config=None
for i,config in enumerate(ParameterSampler(space,n_iter=N_TRIALS,random_state=SEED),1):
    mean,std,runtime=score_cv(config)
    if mean>best:
        best=mean;best_config=config.copy()
    trials.append({'trial':i,**config,'cv_f1_mean':mean,'cv_f1_std':std,'runtime_seconds':runtime,'best_so_far':best,'seed':SEED})
    pd.DataFrame(trials).to_csv(OUT/'trial_history.csv',index=False)
    print(f'{i:02d}/{N_TRIALS} CV macro-F1 {mean:.4f}; best {best:.4f}',flush=True)

def final_eval(label,config):
    model=RandomForestClassifier(**config,random_state=SEED,n_jobs=1)
    t=time.perf_counter();model.fit(X_train,y_train);fit=time.perf_counter()-t
    t=time.perf_counter();pred=model.predict(X_test);predict=time.perf_counter()-t
    return {'method':label,'test_macro_f1':f1_score(y_test,pred,average='macro'),'test_accuracy':accuracy_score(y_test,pred),'fit_seconds':fit,'predict_seconds':predict},pred

base_final,base_preds=final_eval('Default Random Forest',baseline_params)
best_final,best_preds=final_eval('Random Search best CV',best_config)
comparison=pd.DataFrame([{'method':'Default Random Forest','cv_macro_f1':base_mean,'cv_std':base_std,'cv_time_seconds':base_search_time,**{k:v for k,v in base_final.items() if k!='method'}},{'method':'Random Search best CV','cv_macro_f1':best,'cv_std':pd.DataFrame(trials).loc[pd.DataFrame(trials)['cv_f1_mean'].idxmax(),'cv_f1_std'],'cv_time_seconds':sum(t['runtime_seconds'] for t in trials),**{k:v for k,v in best_final.items() if k!='method'}}])
comparison.to_csv(OUT/'comparison.csv',index=False)
with (OUT/'best_config.json').open('w',encoding='utf-8') as f: json.dump(best_config,f,indent=2)
with (OUT/'experiment_config.json').open('w',encoding='utf-8') as f: json.dump({'seed':SEED,'n_trials':N_TRIALS,'cv_folds':CV_FOLDS,'holdout_test_fraction':.2,'scoring':'f1_macro','dataset':'sklearn.datasets.load_breast_cancer','python':platform.python_version(),'scikit_learn':sklearn_version,'search_space':space},f,indent=2)
with (OUT/'classification_report.txt').open('w',encoding='utf-8') as f: f.write('Base\n'+classification_report(y_test,base_preds)+'\nOptimized\n'+classification_report(y_test,best_preds))
pd.DataFrame(confusion_matrix(y_test,best_preds)).to_csv(OUT/'confusion_matrix.csv',index=False,header=False)

hist=pd.DataFrame(trials)
plt.figure(figsize=(8,4.6));plt.plot(hist.trial,hist.best_so_far,label='Best CV macro-F1 so far',linewidth=2);plt.scatter(hist.trial,hist.cv_f1_mean,s=15,alpha=.4,label='Trial CV macro-F1');plt.xlabel('Trial');plt.ylabel('Macro-F1');plt.title('Random Search — Search History');plt.legend();plt.grid(alpha=.2);plt.tight_layout();plt.savefig(FIG/'convergence.png',dpi=160);plt.close()
fig,ax=plt.subplots(figsize=(7.5,4.6)); names=['Baseline','Random Search']; x=np.arange(2); width=.32;ax.bar(x-width/2,comparison.cv_macro_f1,width,label='CV macro-F1');ax.bar(x+width/2,comparison.test_macro_f1,width,label='Test macro-F1');ax.set_xticks(x,names);ax.set_ylim(0,1.08);ax.set_ylabel('Macro-F1');ax.legend();ax.set_title('Baseline vs optimized (holdout evaluated once)');ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(FIG/'comparison.png',dpi=160);plt.close(fig)
plt.figure(figsize=(7.5,4.6));plt.scatter(hist.runtime_seconds,hist.cv_f1_mean,alpha=.7);plt.xlabel('CV evaluation time per trial (seconds)');plt.ylabel('CV macro-F1');plt.title('Quality versus evaluation cost');plt.grid(alpha=.2);plt.tight_layout();plt.savefig(FIG/'tradeoff.png',dpi=160);plt.close()

# Small sensitivity study around selected configuration on training CV only
sensitivity=[]
for depth in [3,5,10,20,None]:
    cfg=best_config.copy();cfg['max_depth']=depth
    m,s,t=score_cv(cfg)
    sensitivity.append({'max_depth':str(depth),'cv_macro_f1':m,'cv_std':s,'runtime_seconds':t})
pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity.csv',index=False)

# Assess stability under varying model randomness on the fixed folds/split; no test-set tuning
stability=[]
for method,config in [('Baseline',baseline_params),('Optimized',best_config)]:
    for seed in [11,23,42]:
        m,s,t=score_cv(config,seed=seed)
        stability.append({'method':method,'model_seed':seed,'cv_macro_f1':m,'cv_std':s,'runtime_seconds':t})
pd.DataFrame(stability).to_csv(OUT/'stability.csv',index=False)
print('\nRESULTS\n',comparison.to_string(index=False),flush=True)
print('CONFIG',best_config,flush=True)
