import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt

r = 60
d = (2 + 3 + 0 + 0 + 6 + 0) % 10
q = 1500 + 25*r
n = 5 + r % 4
p = .04 + .01*d
h = 4 + r % 4
budget = 1200 + 10*r
target = 10 - r % 3
calls = [(2000 + 800*k, 350 if k < n-1 else 600, p if k < n-1 else 0) for k in range(n)]
input_expected = sum(i*(1+retry) for i, o, retry in calls)
output_expected = sum(o*(1+retry) for i, o, retry in calls)
monthly_queries = q*(30+4*h)
monthly_input = input_expected*monthly_queries
monthly_output = output_expected*monthly_queries
peak_queries_per_minute = 5*q*.35/120
peak_requests_per_minute = peak_queries_per_minute*(n+(n-1)*p)
peak_input_tpm = peak_queries_per_minute*input_expected
peak_output_tpm = peak_queries_per_minute*output_expected
layers, kv_heads, head_dim, seq_len = 32, 8, 128, 5800
weight_4bit_gib = 8e9*.5/2**30
kv_cache_gib = (2*layers*kv_heads*head_dim*seq_len*2)/2**30
capacity = math.floor((24*.90-weight_4bit_gib)/kv_cache_gib)
peak_sessions = math.ceil(peak_queries_per_minute*target/60)
needed_gpus = math.ceil(peak_sessions/capacity)
gpu_month = .89*720

def api_cost(volume):
    per_query = (4900*.40+255*1.60)/1e6
    return volume*(30+4*h)*per_query

def gpu_cost(volume):
    sessions = volume*5*.35/120*target/60
    return max(1,math.ceil(sessions/capacity))*gpu_month

cross = next(volume for volume in range(1,20000) if gpu_cost(volume)<api_cost(volume))
values = list(range(0,20001,20))
plt.plot(values,[api_cost(v) for v in values],color="black",label="API")
plt.step(values,[gpu_cost(v) for v in values],where="post",color="black",linestyle="--",label="Self-host")
plt.legend(frameon=False)
plt.xlabel("Normal daily queries")
plt.ylabel("Monthly cost in USD")
plt.savefig(Path(__file__).with_name("Q1_Cost_Plot_Recalculated.png"),dpi=160)
plt.close()
