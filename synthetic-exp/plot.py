import numpy as np
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import time
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *
# import gaussian

import csv

save_file_handles = ['result_scale-rbf-n-50-d-10-m-100-N-5000-attack_level-0',
                     'result_scale-rbf-n-50-d-10-m-100-N-5000-attack_level-1',
                     'result_scale-crossmodel-source_m-50-rbf-n-50-d-10-m-100-N-5000-attack_level-0',
                     'result_scale-crossmodel-source_m-50-rbf-n-50-d-10-m-100-N-5000-attack_level-1',
                     'result_scale-crossmodel-source_m-200-rbf-n-50-d-10-m-100-N-5000-attack_level-0',
                     'result_scale-crossmodel-source_m-200-rbf-n-50-d-10-m-100-N-5000-attack_level-1']
to_plot = 0
save_file_handle = save_file_handles[to_plot]
results_path = 'results/' + save_file_handle + '.csv'

scale_list = []
alpha_S_T_list = []
alpha_T_S_list = []
loss_f_S_list = []
loss_f_T_list = []
gamma_S_T_list = []
gamma_T_S_list = []
alpha_gamma_combined_S_T_list = []
alpha_gamma_combined_T_S_list = []
gradient_loss_list = []

lists = [scale_list, alpha_S_T_list, alpha_T_S_list, gamma_S_T_list, gamma_T_S_list, alpha_gamma_combined_S_T_list,
         alpha_gamma_combined_T_S_list, loss_f_S_list, loss_f_T_list, gradient_loss_list]

with open(results_path) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # print(row)
        if len(row) == 0:
            continue
        for i in range(len(lists)):
            lists[i].append(row[i])

# change list to dict
results = {}
for i in range(len(lists)):
    l = lists[i]
    for j in range(len(l[1:])):
        if float(l[1 + j]) < 0.0001:
            l[1 + j] = 0.0
        else:
            l[1 + j] = float(l[1 + j])
    results[l[0]] = l[1:]

length = len(results['scale'])
values = [('alpha_S_T', 'alpha_1(ST)', 'black'), ('alpha_T_S', 'alpha_T_S', 'black'), ('gamma_S_T', 'gamma_S_T', pal[2]),
        ('gamma_T_S', 'gamma_T_S', pal[2]), ('loss_f_S', 'loss_f_S', pal[3]), ('gradient_loss', 'gradient_loss', pal[4])]

plot_every = 1
marker_plot_every = 2
marker_size = 25
is_forward_KL = False

figs = []

print('Plotting ..')
fig = bkp.figure(y_axis_label='', x_axis_label='scale of perturbation', plot_width=1000, plot_height=1000)
preprocess_plot(fig, '32pt', False, False)
figs.append([fig])

for value in values:

    if value[0] == 'alpha_S_T' or value[0] == "gamma_S_T":
        fig.line(results['scale'], results[value[0]], color=value[2], legend_label=value[1], line_width=10, line_dash="6 6")
    else:
        fig.line(results['scale'], results[value[0]], color=value[2], legend_label=value[1], line_width=10)

    fig.legend.location = 'top_right'

for f in [fig]:
    f.legend.label_text_font_size = '25pt'
    f.legend.glyph_width = 40
    # f.legend.glyph_height = 40
    f.legend.spacing = 15
    f.legend.visible = False
    f.legend.location = 'top_right'

legend_len = len(fig.legend.items)
fig.legend.items = fig.legend.items[legend_len - 2:legend_len] + fig.legend.items[0:legend_len - 2]

postprocess_plot(fig, '22pt', location='top_right', glyph_width=40)
fig.legend.background_fill_alpha = 0.
fig.legend.border_line_alpha = 0.

bkp.show(fig)
#bkp.show(bkl.gridplot(figs))
