"""Exact non-dominated (ride intervals, line changes) cost screening.

This module is a structural diagnostic, NOT a service-aware route universe.
One representative and a tied-path count are retained for every cost label.
Pruned/absent routes have not been assigned zero passenger probability.
"""
from __future__ import annotations
from collections import defaultdict
from heapq import heappop, heappush
from typing import Iterable

Cost = tuple[int, int]

def dominates(a: Cost, b: Cost) -> bool:
    return a != b and a[0] <= b[0] and a[1] <= b[1]

def solve_frontier(adj: dict[str, list[tuple[str,int,int]]], sources: Iterable[str]):
    """Enumerate all non-dominated cost vectors to every reachable node.

    Nonnegative integer edges must have positive L1 norm. Multiplicity counts
    distinct line-state paths, not people, route probabilities or truth.
    """
    labels: dict[str, dict[Cost,dict]] = {u: {} for u in adj}
    heap = []
    for u in sorted(set(sources)):
        if u not in adj: raise ValueError(f'Unknown source node: {u}')
        labels[u][(0,0)] = {'count':1, 'path':(u,), 'parents':[]}
        heappush(heap,(0,0,0,u))
    while heap:
        _, ride, changes, u = heappop(heap)
        rec = labels[u].get((ride,changes))
        if rec is None: continue
        for v,dr,dt in adj[u]:
            if dr<0 or dt<0 or dr+dt<=0: raise ValueError('Invalid edge weight')
            key=(ride+dr, changes+dt)
            old=labels[v]
            if key in old:
                old[key]['count'] += rec['count']
                old[key]['parents'].append((u,(ride,changes)))
                continue
            if any(dominates(k,key) for k in old): continue
            for k in [k for k in old if dominates(key,k)]: del old[k]
            old[key]={'count':rec['count'],'path':rec['path']+(v,), 'parents':[(u,(ride,changes))]}
            heappush(heap,(sum(key),*key,v))
    return labels

def target_frontier(labels, targets: Iterable[str]):
    """Aggregate line-state labels into physical endpoint cost alternatives."""
    result={}
    for t in sorted(set(targets)):
        for key,rec in labels.get(t,{}).items():
            if key in result:
                result[key]['count'] += rec['count']
                if rec['path'] < result[key]['path']: result[key]['path']=rec['path']
            else: result[key]={'count':rec['count'],'path':rec['path']}
    return {k:result[k] for k in sorted(result) if not any(dominates(j,k) for j in result)}

def build_adjacency(nodes, edges):
    adj={u:[] for u in sorted(nodes)}
    seen=set()
    for a,b,kind in edges:
        if a==b or a not in adj or b not in adj: raise ValueError('Invalid endpoint')
        if kind not in {'ride','transfer'}: raise ValueError('Unknown edge kind')
        key=(min(a,b),max(a,b),kind)
        if key in seen: continue
        seen.add(key)
        r,t=(1,0) if kind=='ride' else (0,1)
        adj[a].append((b,r,t));adj[b].append((a,r,t))
    for u in adj: adj[u].sort()
    return adj


def enumerate_efficient_paths(labels, targets):
    """Retain every cost-tied efficient path, without a K or transfer cap.

    This does not enumerate dominated paths that may be service-time competitive.
    Parent chains decrease the positive integer L1 cost, so recursion is avoided.
    """
    front=target_frontier(labels,targets)
    for key,rec in front.items():
        paths=[]
        for target in sorted(set(targets)):
            if key not in labels.get(target,{}): continue
            stack=[(target,key,(target,))]
            while stack:
                node,cost,suffix=stack.pop()
                label=labels[node][cost]
                if not label['parents']:
                    paths.append(suffix)
                else:
                    for parent,pcost in label['parents']:
                        stack.append((parent,pcost,(parent,)+suffix))
        paths.sort()
        if len(paths)!=rec['count'] or len(set(paths))!=len(paths):
            raise AssertionError('Tied-route enumeration/count mismatch')
        rec['paths']=paths
    return front
