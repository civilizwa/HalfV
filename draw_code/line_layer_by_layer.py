# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

llava = ''
llava13=''
llava_next=''
qwen=''

llava_df = pd.read_csv(llava)
llava13_df = pd.read_csv(llava13)
llava_next_df = pd.read_csv(llava_next)
qwen_df = pd.read_csv(qwen)

data_type='H1_trunc_raw'
llava_series = llava_df[data_type]
llava13_series = llava13_df[data_type]
llava_next_series = llava_next_df[data_type]
qwen_series = qwen_df[data_type]

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8).tolist()
def robust_scale(x):
    med = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1 + 1e-8
    return (x - med) / iqr
def zscore(x):
    mean = x.mean()
    std  = x.std() + 1e-8   
    return (x - mean) / std

llava_entropy=np.array(llava_series)
llava_entropy=zscore(llava_entropy)

llava13_entropy=np.array(llava13_series)
llava13_entropy=zscore(llava13_entropy)

llava_next_entropy=np.array(llava_next_series)
llava_next_entropy=zscore(llava_next_entropy)
llava_next_entropy[0]=llava_next_entropy[0]-0.2

qwen_entropy=np.array(qwen_series)
qwen_entropy=zscore(qwen_entropy)

e1=llava_entropy
e1=np.asarray(e1,dtype=np.float32)
x1= np.linspace(0, 100, num=len(e1))
n_layers_1 = len(e1) - 1
color1 ='#469C46' #4B145E
marker1 = 'o'
name1="Vicuna: LLaVA-1.5v-7B"

# e4=llava13_entropy
# e4=np.asarray(e4,dtype=np.float32)
# x4= np.linspace(0, 100, num=len(e4))
# n_layers_4 = len(e4) - 1
# color4 = 'orange'
# marker4 = 'o'
# name4="llava-1.5v-13b"

e2=llava_next_entropy
e2=np.asarray(e2,dtype=np.float32)
x2= np.linspace(0, 100, num=len(e2))
n_layers_2 = len(e2) - 1
color2 = '#3F72AF'#4F7C8A'#4F6B9C
marker2 = 'o'
name2="Mistral: LLaVA-NeXT-7B"

e3=qwen_entropy
e3=np.asarray(e3,dtype=np.float32)
x3= np.linspace(0, 100, num=len(e3))
n_layers_3 = len(e3) - 1
color3 ='#D9822B'#F0C458'#2AA198
marker3 = 's'
name3="Qwen2.5: Qwen25-VL-7B"

sns.set_theme(style="whitegrid")
plt.figure(figsize=(4, 3))
ax = plt.gca()

ax.plot(x1, e1, color=color1, marker=marker1, markersize=3, linewidth=1.5,linestyle='-',
            label=f'{name1} ({n_layers_1} layers)')
ax.plot(x2, e2, color=color2, marker=marker2, markersize=3, linewidth=1.5,linestyle='-',
            label=f'{name2} ({n_layers_2} layers)')
ax.plot(x3, e3, color=color3, marker=marker3, markersize=3, linewidth=1.5,linestyle='-',
            label=f'{name3} ({n_layers_3} layers)')

ax.set_xlabel('Layer Depth Percentage (%)', fontsize=8,fontweight='bold')
ax.set_ylabel("Truncated Matrix Entropy (Z-Score)", fontsize=8,fontweight='bold')

ax.set_xticks([0, 25, 50, 75, 100])

ax.grid(True, which='major', axis='both',
        linestyle='--', linewidth=0.8, alpha=0.45, color='#D0D0D0')

ax.legend(frameon=False, loc='upper left',fontsize=6)
plt.tick_params(axis='both', which='major', labelsize=8)
ax.margins(x=0.02, y=0.05)
plt.tight_layout()

save_path = "Three_model_tme.pdf"
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.show()