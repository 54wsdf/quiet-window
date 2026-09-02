#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json
import geopandas as gpd
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance

SEED=20260902
BETA=0.08
PERIODS=[('night',0,6,.12),('am_peak',6,10,.20),('midday',10,16,.28),('pm_peak',16,20,.22),('evening',20,24,.18)]

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def hav(lat,lon):
    R=6371.0088; p=np.radians(lat); q=np.radians(lon)
    dp=p[None,:]-p[:,None]; dl=q[None,:]-q[:,None]
    a=np.sin(dp/2)**2+np.cos(p[:,None])*np.cos(p[None,:])*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))

def nmae(p,t,m):
    d=np.abs(t[m]).sum()
    return float(np.abs(p[m]-t[m]).sum()/d) if d else None

def main():
    a=argparse.ArgumentParser()
    for x in ['tlc','zones','rac','wac','xwalk','out']: a.add_argument('--'+x,required=True)
    a=a.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    z=gpd.read_file('zip://'+a.zones)
    idc=next(c for c in ['LocationID','locationid','OBJECTID'] if c in z.columns)
    z=z.rename(columns={idc:'LocationID'}).to_crs(4326)
    z['LocationID']=pd.to_numeric(z.LocationID,errors='coerce')
    z=z.dropna(subset=['LocationID','geometry']).copy(); z.LocationID=z.LocationID.astype(int)

    x=pd.read_csv(a.xwalk,compression='gzip',dtype=str)
    bc=next(c for c in ['tabblk2020','tabblk2010','tabblk2000'] if c in x.columns)
    la=next(c for c in ['blklatdd','blklat','lat'] if c in x.columns)
    lo=next(c for c in ['blklondd','blklon','lon'] if c in x.columns)
    x=x[[bc,la,lo]].copy(); x[la]=pd.to_numeric(x[la],errors='coerce'); x[lo]=pd.to_numeric(x[lo],errors='coerce')
    x=x.dropna(); x=x[x[la].between(40.45,40.95)&x[lo].between(-74.35,-73.65)]
    pts=gpd.GeoDataFrame(x[[bc]],geometry=gpd.points_from_xy(x[lo],x[la]),crs=4326)
    bm=gpd.sjoin(pts,z[['LocationID','geometry']],predicate='within',how='inner')[[bc,'LocationID']].drop_duplicates(bc).rename(columns={bc:'block'})
    bm.block=bm.block.astype(str)

    rac=pd.read_csv(a.rac,compression='gzip',dtype={'h_geocode':str})[['h_geocode','C000']].rename(columns={'h_geocode':'block','C000':'P'})
    wac=pd.read_csv(a.wac,compression='gzip',dtype={'w_geocode':str})[['w_geocode','C000']].rename(columns={'w_geocode':'block','C000':'A'})
    for d,c in [(rac,'P'),(wac,'A')]:
        d.block=d.block.astype(str); d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0)
    P=rac.merge(bm,on='block').groupby('LocationID',as_index=False).P.sum()
    A=wac.merge(bm,on='block').groupby('LocationID',as_index=False).A.sum()
    act=P.merge(A,on='LocationID',how='outer').fillna(0); act['activity']=act.P+act.A
    sel=act[act.activity>0].sort_values(['activity','LocationID'],ascending=[False,True]).head(200).LocationID.astype(int).tolist()
    if len(sel)!=200: raise SystemExit(f'only {len(sel)} activity zones')
    act=act.set_index('LocationID').reindex(sel).fillna(0)
    zv=z.set_index('LocationID').reindex(sel)
    cent=gpd.GeoSeries(zv.to_crs(2263).geometry.centroid,crs=2263).to_crs(4326)
    lat=cent.y.to_numpy(); lon=cent.x.to_numpy(); dist=hav(lat,lon)
    pp=act.P.to_numpy(float); aa=act.A.to_numpy(float)
    prior=np.outer(pp,aa)*np.exp(-BETA*dist); np.fill_diagonal(prior,0); prior/=prior.sum()

    t=pd.read_parquet(a.tlc,columns=['tpep_pickup_datetime','PULocationID','DOLocationID'])
    t['dt']=pd.to_datetime(t.tpep_pickup_datetime,errors='coerce')
    t.PULocationID=pd.to_numeric(t.PULocationID,errors='coerce'); t.DOLocationID=pd.to_numeric(t.DOLocationID,errors='coerce')
    t=t.dropna(subset=['dt','PULocationID','DOLocationID'])
    t=t[(t.dt>='2026-05-01')&(t.dt<'2026-06-01')]; t.PULocationID=t.PULocationID.astype(int); t.DOLocationID=t.DOLocationID.astype(int)
    t=t[t.PULocationID.isin(sel)&t.DOLocationID.isin(sel)&(t.PULocationID!=t.DOLocationID)].copy()
    t['period']=''
    for name,l,h,_ in PERIODS: t.loc[(t.dt.dt.hour>=l)&(t.dt.dt.hour<h),'period']=name
    idx={x:i for i,x in enumerate(sel)}
    truth={}
    for name,_,_,share in PERIODS:
        m=np.zeros((200,200))
        g=t[t.period==name].groupby(['PULocationID','DOLocationID']).size()
        for (o,d),v in g.items(): m[idx[int(o)],idx[int(d)]]=v
        truth[name]=(m,share)

    rows=[]; cells=[]; raw_total=float(pp.sum())
    for name,_,_,share in PERIODS:
        y=truth[name][0]; raw=prior*raw_total*share; shape=raw*(y.sum()/raw.sum()) if y.sum()>0 else raw
        mask=~np.eye(200,dtype=bool)
        pe=np.abs(shape.sum(1)-y.sum(1)).sum()/max(y.sum(),1); ae=np.abs(shape.sum(0)-y.sum(0)).sum()/max(y.sum(),1)
        wd=wasserstein_distance(dist.ravel(),dist.ravel(),u_weights=np.clip(shape.ravel(),0,None),v_weights=np.clip(y.ravel(),0,None)) if y.sum()>0 else None
        rows.append({'period':name,'truth_taxi_trips':float(y.sum()),'raw_pred_worker_units':float(raw.sum()),'shape_normalized_od_nmae':nmae(shape,y,mask),'shape_normalized_pa_nmae':float((pe+ae)/2),'trip_length_wasserstein_km':None if wd is None else float(wd)})
        for i,o in enumerate(sel):
            for j,d in enumerate(sel):
                if i!=j: cells.append((name,o,d,y[i,j],raw[i,j],shape[i,j],dist[i,j]))
    pd.DataFrame(rows).to_csv(out/'t6_tlc_period_metrics.csv',index=False)
    pd.DataFrame(cells,columns=['period','origin_zone','destination_zone','truth_taxi_trips','raw_model_worker_units','shape_normalized_prediction','distance_km']).to_csv(out/'t6_tlc_od_cells.csv',index=False)

    y=sum((v[0] for v in truth.values()),np.zeros((200,200))); spatial=prior.copy(); non=np.outer(pp,aa); np.fill_diagonal(non,0)
    met=[]; rcells=[]
    for rate in [.05,.10,.20]:
        rng=np.random.default_rng(SEED+int(rate*1000)); obs=np.zeros(y.size,dtype=bool); obs[rng.choice(y.size,size=round(rate*y.size),replace=False)]=True; obs=obs.reshape(y.shape); np.fill_diagonal(obs,False)
        un=~obs; np.fill_diagonal(un,False)
        for method,base in [('least_squares_activity_distance_prior',spatial),('least_squares_nonspatial_activity_prior',non)]:
            den=np.dot(base[obs],base[obs]); sc=float(np.dot(base[obs],y[obs])/den) if den else 0.; p=base*sc; p[obs]=y[obs]; np.fill_diagonal(p,0)
            met.append({'observation_rate':rate,'method':method,'observed_cells':int(obs.sum()),'unobserved_cells':int(un.sum()),'fit_scale_from_observed_cells_only':sc,'unobserved_od_nmae':nmae(p,y,un),'row_margin_nmae':float(np.abs(p.sum(1)-y.sum(1)).sum()/max(y.sum(),1)),'column_margin_nmae':float(np.abs(p.sum(0)-y.sum(0)).sum()/max(y.sum(),1)),'predicted_total_trips':float(p.sum()),'truth_total_trips':float(y.sum())})
            if method.startswith('least_squares_activity'):
                for i,o in enumerate(sel):
                    for j,d in enumerate(sel):
                        if i!=j: rcells.append((rate,o,d,int(obs[i,j]),y[i,j],p[i,j]))
    pd.DataFrame(met).to_csv(out/'t7_tlc_sparse_od_metrics.csv',index=False)
    pd.DataFrame(rcells,columns=['observation_rate','origin_zone','destination_zone','is_observed','truth_trips','reconstructed_trips']).to_csv(out/'t7_tlc_sparse_od_cells.csv',index=False)
    pd.DataFrame({'LocationID':sel,'centroid_lat':lat,'centroid_lon':lon,'lodes_resident_workers_C000':pp,'lodes_workplace_jobs_C000':aa}).to_csv(out/'zone_activity_and_centroids.csv',index=False)

    r=pd.DataFrame(rows)
    summary={'evidence_class':'public_data registry / official public source rehearsal',
      'T6':{'primary_truth_source':'NYC TLC yellow taxi trip records, May 2026','support_input_source':'LEHD LODES NY 2023 RAC/WAC + crosswalk','zone_count':200,'truth_days':31,'truth_mode_scope':'yellow taxi only; not total multimodal demand','model_uses_scored_region_od_observation':False,'fixed_beta_per_km':BETA,'fixed_period_shares':{n:s for n,_,_,s in PERIODS},'raw_prediction_units':'worker-unit proxy; no TLC-derived trip-rate or total fitted','shape_diagnostic_only':{'mean_od_nmae':float(r.shape_normalized_od_nmae.mean()),'mean_pa_nmae':float(r.shape_normalized_pa_nmae.mean()),'mean_trip_length_wasserstein_km':float(r.trip_length_wasserstein_km.mean()),'warning':'equal-total normalization is evaluation-side only and is not a C6-compliant submission transformation'},'formal_status':'PRIMARY_TRUTH_REHEARSAL_CLOSED_WITH_SCOPE_GAPS','remaining_gap':'LODES RAC is employed-resident activity, not full population by segment; TLC yellow taxi is one mode; no organizer hidden split exists.'},
      'T7':{'primary_truth_source':'NYC TLC yellow taxi OD aggregated to taxi zones','observation_rates':[.05,.10,.20],'metrics':met,'formal_status':'SPARSE_OD_SUBPROBLEM_CLOSED','remaining_gap':'NYC TLC alone does not provide path flows, path-link incidence or independent link-count truth; full T7 conservation/count closure remains open.'},
      'selection':{'rule':'top 200 taxi zones by LODES RAC+WAC activity only; TLC truth not consulted','selected_zone_count':200},
      'source_sha256':{'tlc_parquet':sha(a.tlc),'taxi_zones_zip':sha(a.zones),'lodes_rac':sha(a.rac),'lodes_wac':sha(a.wac),'lodes_xwalk':sha(a.xwalk)},
      'provenance_separation':{'official_public_truth':['NYC TLC yellow_tripdata_2026-05.parquet','NYC TLC taxi_zones.zip'],'official_public_support':['LEHD LODES NY 2023 RAC','LEHD LODES NY 2023 WAC','LEHD LODES NY crosswalk'],'teacher_or_asu_data_used':False,'self_added_assumptions':{'fixed_beta_per_km':BETA,'fixed_period_shares':{n:s for n,_,_,s in PERIODS}}},
      'canonical_code':'54wsdf/knowledge-hub/domains/academic-research/projects/open-city-traffic-challenge-rehearsal/code/t6_t7_tlc_primary_public_registry.py'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
