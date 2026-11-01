"""Single-file recovery of the v9 April-19 time33 cache; no raw-log scan."""
from __future__ import annotations
import hashlib, json, pickle, time, zipfile
from pathlib import Path
import numpy as np
import time_channel_v9 as v9

OUT=v9.OUT; NAME='apr19'; SCHEMA='time33'; OLD=OUT/'apr19_time33.corrupt.npz'; FINAL=OUT/'apr19_time33.npz'; TMP=OUT/'apr19_repair_tmp_time33.npz'
def h(a):
    q=hashlib.sha256();q.update(np.ascontiguousarray(a).view(np.uint8));return q.hexdigest()
def main():
    started=time.perf_counter()
    if FINAL.exists() or not OLD.exists(): raise RuntimeError('expected corrupt archive renamed before repair')
    old=v9.identity(OLD); side=json.loads((OUT/'apr19_time33.fingerprint.json').read_text(encoding='utf-8'))
    ids={n:v9.identity(p) for n,p in {'v8_submission':v9.V8/'submission_channel_v8_channel_hgb.csv','v8_validation_model':v9.V8/'channel_hgb_validation.pkl','v8_final_model':v9.V8/'channel_hgb_final.pkl'}.items()}
    base=v9.digest({'v8_inputs':ids,'v8_fp':json.loads((v9.V8/'summary.json').read_text(encoding='utf-8'))['fingerprint'],'feature_schemas':{'time33':v9.V8_FEATURES+v9.TIME_FEATURES,'time35':v9.V8_FEATURES+v9.TIME_FEATURES+v9.RECENT_FEATURES}})
    profile=pickle.load((OUT/'time_channel_profiles.pkl').open('rb'))
    x,y,keys,source_fp=v9.features29(NAME); source_x,source_y,source_keys,_=v9.load_npz(v9.V4/'features_long_apr19_T.npz')
    if keys!=source_keys or y is None or not np.array_equal(y,source_y): raise RuntimeError('authoritative source key/label mismatch')
    timings={}
    if not TMP.exists():
        exposure,days=v9.schedule_exposure(v9.CUTS[NAME],v9.T)
        v9.add_features(x,keys,profile,exposure,days,SCHEMA,base,NAME,timings,output_name='apr19_repair_tmp')
    with zipfile.ZipFile(TMP) as z: names=sorted(z.namelist())
    with np.load(TMP,allow_pickle=False) as z:
        repaired=z['X'];fp=str(z['fingerprint'])
    expected_data={'base':base,'name':NAME,'schema':SCHEMA,'features':v9.V8_FEATURES+v9.TIME_FEATURES,'rows_digest':v9.digest(keys)};expected=v9.digest(expected_data)
    checks={'zip_integrity':names==['X.npy','fingerprint.npy'],'shape':list(repaired.shape),'expected_shape':[884800,33],'shape_match':repaired.shape==(884800,33),'schema_match':side['features']==v9.V8_FEATURES+v9.TIME_FEATURES,'finite':bool(np.isfinite(repaired).all()),'source_key_alignment':keys==source_keys,'source_labels_unchanged':bool(np.array_equal(y,source_y)),'source_first25_match':bool(np.array_equal(repaired[:,:25],source_x[:,:25])),'v8_first29_match':bool(np.array_equal(repaired[:,:29],x)),'fingerprint':fp,'expected_fingerprint':expected,'fingerprint_match':fp==expected,'sidecar_rows_digest_match':side['rows_digest']==expected_data['rows_digest'],'cutoff':'2020-04-19 00:00:00','prediction_window_end':'2020-05-18 00:00:00','no_raw_log_scan':True}
    if not all(checks[k] for k in ('zip_integrity','shape_match','schema_match','finite','source_key_alignment','source_labels_unchanged','source_first25_match','v8_first29_match','fingerprint_match','sidecar_rows_digest_match')): raise RuntimeError(f'repair validation failed: {checks}')
    TMP.replace(FINAL); TMP.with_suffix('.fingerprint.json').unlink(missing_ok=True)
    evidence={'status':'repaired','old':old,'new':v9.identity(FINAL),'cache_sources':{'profile':str((OUT/'time_channel_profiles.pkl').resolve()),'source_matrix':str((v9.V4/'features_long_apr19_T.npz').resolve()),'v8_matrix':str((v9.V8/'apr19_channel29.npz').resolve()),'schedule_window':'2020-04-19 <= airing_start < 2020-05-18'},'validation_checks':checks,'generation_timing':timings,'runtime_sec':time.perf_counter()-started}
    v9.save_json(v9.ROOT/'results/ranker_v10/repair_apr19.json',evidence);print(json.dumps(evidence,indent=2))
if __name__=='__main__': main()
