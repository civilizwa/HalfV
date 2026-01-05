# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

whole_file_name = ''
visual_file_name=''
text_file_name=''

whole_df = pd.read_csv(whole_file_name)
visual_df = pd.read_csv(visual_file_name)
text_df = pd.read_csv(text_file_name)

whole_series = whole_df['H1_trunc_raw']
visual_series = visual_df['H1_trunc_raw']
text_series = text_df['H1_trunc_raw']

whole_entropy=whole_series.tolist()
visual_entropy=visual_series.tolist()
text_entropy=text_series.tolist()

e1=whole_entropy
e1=np.asarray(e1,dtype=np.float32)
x1= np.linspace(0, 100, num=len(e1))
n_layers_1 = len(e1)-1
color1 = 'blue'#4F7C8A'
marker1 = 'o'
name1="Both Modality"

e2=visual_entropy
e2=np.asarray(e2,dtype=np.float32)
x2= np.linspace(0, 100, num=len(e2))
n_layers_2 = len(e2)-1
color2 = 'green'#8FA876'
marker2 = 'o'
name2="Visual Modality"

e3=text_entropy
e3=np.asarray(e3,dtype=np.float32)
x3= np.linspace(0, 100, num=len(e3))
n_layers_3 = len(e3)-1
color3 = 'orange'#F0C458'
marker3 = 'o'
name3="Text Modality"

sns.set_theme(style="whitegrid")
plt.figure(figsize=(4,3))
ax = plt.gca()

ax.plot(x1, e1, color=color1, marker=marker1, markersize=2, linewidth=1.0,linestyle='-',
            label=f'{name1} ({n_layers_1} layers)')
ax.plot(x2, e2, color=color2, marker=marker2, markersize=2, linewidth=1.0,linestyle='-',
            label=f'{name2} ({n_layers_2} layers)')
ax.plot(x3, e3, color=color3, marker=marker3, markersize=2, linewidth=1.0,linestyle='-',
            label=f'{name3} ({n_layers_3} layers)')
ax.set_xlabel('Layer Depth Percentage (%)',fontsize=8,fontweight='bold')
ax.set_ylabel("Truncated Matrix Entropy",fontsize=8,fontweight='bold')

ax.set_xticks([0, 25, 50, 75, 100])

# 开启淡灰色网格
ax.grid(True, which='major', axis='both',
        linestyle='--', linewidth=0.8, alpha=0.45, color='#D0D0D0')

ax.legend(frameon=False, loc='upper left',fontsize=6)
plt.tick_params(axis='both', which='major', labelsize=8)
ax.margins(x=0.02, y=0.05)

plt.tight_layout()

save_path = "three_modality_tme.pdf"
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.show()
