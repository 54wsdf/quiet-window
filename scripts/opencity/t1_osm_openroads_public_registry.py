#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import geopandas as gpd,networkx as nx,numpy as np,osmium,pandas as pd,pyogrio
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString,Point,box
SEED=20260902; BBOX=(-4.45,55.75,-4.05,55.95)
def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def orient(g):
 try:
  if g is None or g.is_empty:return np.nan
  if g.geom_type=='MultiLineString':g=max(g.geoms,key=lambda x:x.length)
  c=list(g.coords);dx=c[-1][0]-c[0][0];dy=c[-1][1]-c[0][1]
  return np.nan if dx==dy==0 else (math.degrees(math.atan2(dy,dx))+180)%180
 except:return np.nan
def odiff(a,b):
 if not np.isfinite(a) or not np.isfinite(b):return np.nan
 d=abs(a-b)%180;return min(d,180-d)
class H(osmium.SimpleHandler):
 def __init__(self,b):super().__init__();self.b=b;self.nodes={};self.edges=[];self.ways=[]
 def way(self,w):
  hw=w.tags.get('highway')
  if not hw or hw in {'footway','path','cycleway','steps','pedestrian','bridleway','construction','proposed'}:return
  seq=[];inside=False
  try:
   for n in w.nodes:
    lon=float(n.lon);lat=float(n.lat);nid=int(n.ref);seq.append((nid,lon,lat));inside|=self.b[0]-.03<=lon<=self.b[2]+.03 and self.b[1]-.03<=lat<=self.b[3]+.03
  except osmium.InvalidLocationError:return
  if not inside or len(seq)<2:return
  for nid,lon,lat in seq:self.nodes[nid]=(lon,lat)
  for a,b in zip(seq,seq[1:]):self.edges.append((a[0],b[0]))
  try:g=LineString([(x[1],x[2]) for x in seq])
  except:return
  self.ways.append({'osm_way_id':int(w.id),'osm_highway':hw,'osm_ref':w.tags.get('ref') or '','osm_name':w.tags.get('name') or '','geometry':g})
def extract_osm(p):
 h=H(BBOX);h.apply_file(str(p),locations=True)
 tf=Transformer.from_crs(4326,27700,always_xy=True);ids=np.array(list(h.nodes),dtype=np.int64);xy=np.array([tf.transform(*h.nodes[int(i)]) for i in ids]);pos={int(i):(float(x),float(y)) for i,(x,y) in zip(ids,xy)}
 G=nx.Graph();G.add_nodes_from((n,{'x':v[0],'y':v[1]}) for n,v in pos.items())
 for a,b in h.edges:
  if a in pos and b in pos and a!=b:
   L=math.dist(pos[a],pos[b]);
   if L>0:G.add_edge(a,b,length_m=L)
 ways=gpd.GeoDataFrame(h.ways,geometry='geometry',crs=4326).to_crs(27700);x0,y0=tf.transform(BBOX[0],BBOX[1]);x1,y1=tf.transform(BBOX[2],BBOX[3]);clip=box(min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1));ways=ways[ways.intersects(clip)].copy();ways['geometry']=ways.geometry.intersection(clip);ways=ways[~ways.geometry.is_empty]
 return G,pos,ways,clip
def layer(gpkg,token):
 names=[str(r[0]) for r in pyogrio.list_layers(gpkg)];hit=[n for n in names if token.lower() in ''.join(c for c in n.lower() if c.isalnum())]
 if not hit:raise RuntimeError(f'{token} layer missing {names}')
 return hit[0]
def bbox_crs(crs):
 tf=Transformer.from_crs(4326,crs,always_xy=True);p=[tf.transform(BBOX[i],BBOX[j]) for i,j in [(0,1),(0,3),(2,1),(2,3)]];xs=[x for x,y in p];ys=[y for x,y in p];return min(xs),min(ys),max(xs),max(ys)
def load_or(gpkg):
 ll,nl=layer(gpkg,'roadlink'),layer(gpkg,'roadnode');li,ni=pyogrio.read_info(gpkg,layer=ll),pyogrio.read_info(gpkg,layer=nl);L=pyogrio.read_dataframe(gpkg,layer=ll,bbox=bbox_crs(li['crs'])).to_crs(27700);N=pyogrio.read_dataframe(gpkg,layer=nl,bbox=bbox_crs(ni['crs'])).to_crs(27700)
 c={x.lower():x for x in L.columns};nc={x.lower():x for x in N.columns};idc,sc,ec,nidc=c.get('id'),c.get('start_node'),c.get('end_node'),nc.get('id')
 if not all([idc,sc,ec,nidc]):raise RuntimeError(f'fields missing L={list(L.columns)} N={list(N.columns)}')
 L=L.rename(columns={idc:'or_link_id',sc:'start_node',ec:'end_node'}).dropna(subset=['or_link_id','start_node','end_node','geometry']);N=N.rename(columns={nidc:'or_node_id'}).dropna(subset=['or_node_id','geometry'])
 for q in ['or_link_id','start_node','end_node']:L[q]=L[q].astype(str)
 N.or_node_id=N.or_node_id.astype(str);pos={r.or_node_id:(float(r.geometry.x),float(r.geometry.y)) for r in N[['or_node_id','geometry']].itertuples(index=False)};G=nx.Graph();G.add_nodes_from(pos)
 for r in L[['start_node','end_node','geometry']].itertuples(index=False):
  if r.start_node in pos and r.end_node in pos and r.start_node!=r.end_node:G.add_edge(r.start_node,r.end_node,length_m=float(r.geometry.length))
 return L,N,G,pos,ll,nl
def cmap(G):
 o={}
 for k,c in enumerate(nx.connected_components(G)):
  for n in c:o[n]=k
 return o
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--osm',required=True,type=Path);ap.add_argument('--openroads',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--audit-links',type=int,default=200);ap.add_argument('--od-pairs',type=int,default=500);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 Go,op,ow,clip=extract_osm(a.osm);L,N,Gt,tp,ll,nl=load_or(a.openroads);L=L[L.intersects(clip)].copy();N=N[N.intersects(clip)].copy();ow['oo']=ow.geometry.map(orient);L['ro']=L.geometry.map(orient)
 m=gpd.sjoin_nearest(L[['or_link_id','start_node','end_node','ro','geometry']],ow[['osm_way_id','osm_highway','osm_ref','osm_name','oo','geometry']],how='left',max_distance=100,distance_col='nearest_osm_distance_m');m['orientation_diff_deg']=[odiff(x,y) for x,y in zip(m.ro,m.oo)];m=m.sort_values(['or_link_id','nearest_osm_distance_m','orientation_diff_deg']).drop_duplicates('or_link_id');m['well_represented']=(m.nearest_osm_distance_m<=20)&(m.orientation_diff_deg<=30);cand=m[~m.well_represented].copy().sort_values(['nearest_osm_distance_m','orientation_diff_deg','or_link_id'],na_position='last')
 oi=np.array(list(op));oxy=np.array([op[int(i)] for i in oi]);tree=cKDTree(oxy);tn=N.set_index('or_node_id').geometry.to_dict();Gr=Go.copy();rep=[]
 for r in cand.head(10000).itertuples(index=False):
  if r.start_node not in tn or r.end_node not in tn:continue
  p,q=tn[r.start_node],tn[r.end_node];d1,i1=tree.query([p.x,p.y]);d2,i2=tree.query([q.x,q.y]);u,v=int(oi[i1]),int(oi[i2])
  if d1<=60 and d2<=60 and u!=v and not Gr.has_edge(u,v):
   leng=float(r.geometry.length);Gr.add_edge(u,v,length_m=max(leng,1));rep.append({'or_link_id':r.or_link_id,'osm_u':u,'osm_v':v,'start_snap_m':float(d1),'end_snap_m':float(d2),'or_length_m':leng})
 audit=cand.head(a.audit_links)[['or_link_id','start_node','end_node','nearest_osm_distance_m','orientation_diff_deg','well_represented']].copy();audit['audit_truth']='OS_Open_Roads_public_reference_link';audit.to_csv(a.output/'topology_reference_audit_200.csv',index=False);pd.DataFrame(rep).to_csv(a.output/'repair_proposals.csv',index=False);m[['or_link_id','osm_way_id','osm_highway','nearest_osm_distance_m','orientation_diff_deg','well_represented']].to_csv(a.output/'openroads_osm_conflation.csv',index=False)
 x0,y0,x1,y1=clip.bounds;xs=np.linspace(x0+(x1-x0)/10,x1-(x1-x0)/10,5);ys=np.linspace(y0+(y1-y0)/10,y1-(y1-y0)/10,5);zxy=np.array([(x,y) for y in ys for x in xs]);zid=[f'Z{i:02d}' for i in range(25)];ti=np.array(list(tp),dtype=object);txy=np.array([tp[x] for x in ti]);tt=cKDTree(txy);td,tj=tt.query(zxy);od,oj=tree.query(zxy);Z=pd.DataFrame({'zone_id':zid,'x_27700':zxy[:,0],'y_27700':zxy[:,1],'truth_node':[ti[i] for i in tj],'truth_snap_m':td,'osm_node':[int(oi[i]) for i in oj],'osm_snap_m':od});Z.to_csv(a.output/'self_added_zone_grid.csv',index=False)
 ct,co,cr=cmap(Gt),cmap(Go),cmap(Gr);pairs=[]
 for i in range(25):
  for j in range(25):
   if i==j:continue
   A,B=Z.iloc[i],Z.iloc[j]
   if ct.get(A.truth_node)==ct.get(B.truth_node):pairs.append({'origin':A.zone_id,'destination':B.zone_id,'truth_route':True,'osm_raw_route':co.get(int(A.osm_node))==co.get(int(B.osm_node)),'osm_repaired_route':cr.get(int(A.osm_node))==cr.get(int(B.osm_node))})
 rng=np.random.default_rng(SEED);rng.shuffle(pairs);pairs=pairs[:a.od_pairs];P=pd.DataFrame(pairs);P.to_csv(a.output/'routing_audit_500_pairs.csv',index=False)
 if len(P)<a.od_pairs:raise RuntimeError(f'only {len(P)} truth-connected pairs')
 S={'task':'T1 topology correction','evidence_class':'public_data registry / official public source rehearsal','primary_source':'OpenStreetMap Scotland Geofabrik PBF','reference_source':'Ordnance Survey OS Open Roads April 2026 GeoPackage','study_area':'Glasgow urban bbox','study_bbox_lonlat':list(BBOX),'source_sha256':{'osm_scotland_pbf':sha256(a.osm),'os_open_roads_gpkg':sha256(a.openroads)},'openroads_layers':{'road_link':ll,'road_node':nl},'scale':{'osm_graph_nodes':Go.number_of_nodes(),'osm_graph_edges':Go.number_of_edges(),'osm_way_features':len(ow),'openroads_graph_nodes':Gt.number_of_nodes(),'openroads_graph_edges':Gt.number_of_edges(),'openroads_links_in_window':len(L),'conflated_openroads_links':len(m)},'conflation':{'well_represented_links':int(m.well_represented.sum()),'well_represented_share':float(m.well_represented.mean()),'candidate_reference_gaps':int((~m.well_represented).sum()),'distance_threshold_m':20,'orientation_threshold_deg':30,'median_nearest_osm_distance_m':float(m.nearest_osm_distance_m.dropna().median()),'p95_nearest_osm_distance_m':float(m.nearest_osm_distance_m.dropna().quantile(.95))},'rule_based_repair':{'proposed_and_added_edges':len(rep),'endpoint_snap_threshold_m':60,'candidate_scan_cap':10000},'audit':{'public_reference_links':len(audit),'manual_hand_audit_available':False},'routing':{'zone_layer':'self-added deterministic 5x5 grid','zone_count':25,'truth_connected_od_pairs_scored':len(P),'raw_osm_route_success':float(P.osm_raw_route.mean()),'repaired_osm_route_success':float(P.osm_repaired_route.mean()),'route_success_delta':float(P.osm_repaired_route.mean()-P.osm_raw_route.mean())},'status':'PUBLIC_OSM_OPENROADS_TOPOLOGY_REHEARSAL_CLOSED','formal_boundary':'OS Open Roads supplies an independent public network reference, not organizer hand-audited topology-error truth. Turn restrictions, official OD travel-time truth and same-source manual audit labels remain unavailable.','provenance_separation':{'teacher_or_asu_data_used':False,'self_added_data_used':False,'self_added_assumptions':{'Glasgow_bbox':list(BBOX),'zone_grid':'5x5 deterministic','conflation_threshold_m':20,'orientation_threshold_deg':30,'repair_snap_threshold_m':60,'seed':SEED}}};(a.output/'summary.json').write_text(json.dumps(S,indent=2)+'\n');print(json.dumps(S,indent=2))
if __name__=='__main__':main()
