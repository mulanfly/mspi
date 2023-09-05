import json
import os
import git
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple, List


class OperatorAutomaticPartitioner:
    def __init__(self, 
                 model_name: str, 
                 t_call: float=0.01,
                 m_call: int=800 * 2**20,
                 network_bandwidth: float=1 * 2**30):
        # network_bandwidth is bps, need to convert to Bps
        network_bandwidth = network_bandwidth / 8

        repo = git.Repo('.', search_parent_directories=True)
        repo.working_tree_dir

        with open(os.path.join(repo.working_tree_dir, 'configs/profiling_config.json'), 'r') as f:
            profiling_config = json.load(f)
        
        lp = pd.read_csv(os.path.join(profiling_config["profiling_root"], profiling_config["layer_filename"]))
        lp = lp.loc[lp['model_name'].str.contains(model_name)].sort_values(by=['model_name', 'layer_id'])
        
        # check if multiple wholegraphs existed rather than one
        if len(lp['model_name'].unique()) > 1:
            lp = lp.groupby(['model_name', 'layer_id', 'layer_name']).aggregate(
                {'param_size': 'mean', 'exec_time': 'mean', 'pass_size': 'mean'}).reset_index()
            lp['layer_id'] = list(range(1, len(lp['layer_id'])+1))
            lp = lp.groupby(['layer_id']).aggregate(
                {'param_size': 'sum', 'exec_time': 'sum', 'pass_size': 'sum'})
            
            self.m = [None] + [2*i for i in lp.param_size.values.tolist()]
            self.t = [None] + lp.exec_time.values.tolist()
            self.L = len(self.m) - 1

            wp = pd.read_csv(os.path.join(profiling_config["profiling_root"], profiling_config["wholegraph_filename"]))
            wp = wp.loc[(wp['model_name'].str.contains(model_name))
                        & (wp['device_type'] == 'cuda')]
            wp = wp.groupby('model_name').aggregate({'load_time': 'mean'}).aggregate({'load_time': 'sum'})
            
        # only single wholegraph
        else:
            lp = lp.groupby('layer_id').aggregate(
                {'param_size': 'mean', 'exec_time': 'mean', 'pass_size': 'mean'})
            self.m = [None] + [2*i for i in lp.param_size.values.tolist()]
            self.t = [None] + lp.exec_time.values.tolist()
            self.L = len(self.m) - 1

            wp = pd.read_csv(os.path.join(profiling_config["profiling_root"], profiling_config["wholegraph_filename"]))
            wp = wp.loc[(wp['model_name'] == model_name)
                        & (wp['device_type'] == 'cuda')]
            wp = wp.aggregate({'load_time': 'mean'})

        self.t_exec: float = sum(self.t[1:])
        self.m_total: int = sum(self.m[1:])
        self.m_pass: int = int(lp.pass_size.mean())
        self.t_pass: float = self.m_pass / network_bandwidth
        self.m_call: int = m_call
        self.t_call: float = t_call
        self.t_load: float = wp.load_time
        
        self.m_min = None
        self.dp = None
        self.par = None
        self.dp_latency = None
        self.par_latency = None

    def M(self, i: int, j: int):
        return self.m_pass + self.m_call + sum(self.m[k] for k in range(i, j+1))

    def R(self, i: int, j: int):
        return self.t_pass + self.t_call + sum(self.t[k] for k in range(i, j+1))

    def T_resp(self, n: int):
        return n * (self.t_call + self.t_pass) + self.t_exec

    def F(self, N: int, m_min: int):
        dp = [[float('inf')] * (self.L+1) for _ in range(self.L+1)]
        par = [[0] * (self.L+1) for _ in range(self.L+1)]
        mm = self.m_pass + self.m_call
        rr = self.t_pass + self.t_call
        for i in range(1, self.L+1):
            mm += self.m[i]
            rr += self.t[i]
            dp[1][i] = rr
        for g in tqdm(range(2, N+1)):
            for h in range(1, self.L+1):
                mm = self.m_pass + self.m_call
                rr = self.t_pass + self.t_call
                for i in reversed(range(1, h+1)):
                    mm += self.m[i]
                    rr += self.t[i]
                    if mm >= m_min:
                        d = max(dp[g-1][i-1], rr)
                        if dp[g][h] >= d:
                            dp[g][h] = d
                            par[g][h] = i-1
        self.dp_latency = dp
        self.par_latency = par
        return dp, par

    def P(self, SLO: float, m_min: int, n: int) -> Tuple[List[int], float, int]:
        N = min(int(self.M(1, self.L) // m_min), n)
        dp, par = self.F(N, m_min)
        self.m_min = m_min
        
        # maximize q
        min_g, max_q = 0, 0
        for g in range(1, N+1):
            latency = dp[g][self.L]
            t_cold = self.t_load / g + self.T_resp(g)
            if t_cold > SLO:
                continue
            q = int((SLO - self.T_resp(g)) // latency) + 1
            if max_q <= q:
                min_g = g
                max_q = q
        if min_g == 0:
            min_g = N
        g = min_g
        l = self.L
        latency = dp[g][l]
        cutpoints = []
        while l > 0:
            cutpoints.append(l)
            l = par[g][l]
            g -= 1
        q = int((SLO - self.T_resp(min_g)) // latency) + 1
        return cutpoints, latency, q