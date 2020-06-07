import matplotlib
matplotlib.use('Agg')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

with open('../results/affinities/all_affinities.pkl','rb') as file:
    all_affinities = pickle.load(file)
with open('../results/affinities/all_affinities_16k.pkl','rb') as file:
    all_affinities_16k = pickle.load(file)
with open('../results/affinities/all_affinities_1k.pkl', 'rb') as file:
    all_affinities_1k = pickle.load(file)

list_of_tasks = 'autoencoder curvature edge2d keypoint2d segment2d edge3d keypoint3d reshade rgb2depth rgb2mist rgb2sfnorm segment25d segmentsemantic class_1000 class_places'

list_of_tasks_complete = 'Autoencoder Curvature Edge-2D Keypoint-2D Segmentation-2D Edge-3D Keypoint-3D Reshade Depth Euclidean-Distance Surface-Normal Segmentation-3D Segmentation-Semantic Object-Classification Scene-Classification'

list_of_tasks = list_of_tasks.split()
list_of_tasks_complete = list_of_tasks_complete.split()
num_task = len(list_of_tasks)
gt_affinities = np.zeros(shape = (num_task, num_task))
for i,t1 in enumerate(list_of_tasks):
    if t1 == 'segmentsemantic':
        t1 = 'segmentsemantic_rb'
    if t1 == 'vanishing_point':
        t1 = 'vanishing_point_well_defined'
    if t1 == 'class_places':
        t1 = 'class_selected'
    for j,t2 in enumerate(list_of_tasks):
        if t2 == 'segmentsemantic':
            t2 = 'segmentsemantic_rb'
        if t2 == 'vanishing_point':
            t2 = 'vanishing_point_well_defined'
        if t2 == 'class_places':
            t2 = 'class_selected'
        gt_affinities[i][j] = all_affinities[t2 + '__' + t1]

def compete(loss1, loss2):
    count = 0
    total = len(loss1) 
    for i in range(total):
        if loss1[i] > loss2[i] * 1.001:
            count+=1
    return np.clip(count/total, 0.001, 0.999)

def viz(data,path, legend = False):
    plt.figure(figsize=(1,1))
    pal = ['#D08892', '#7AA772', '#7C9FC5']
    print(pal)
    colors = [pal[0] for _ in range(5)] + [pal[1] for _ in range(7)] + [pal[2] for _ in range(3)]
    fig = plt.figure(figsize=(10, 4))
    
    
    model = AgglomerativeClustering(n_clusters=None,distance_threshold=0)
    model = model.fit(data.T)
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    Z = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    
    dn = dendrogram(Z, leaf_font_size = 20, color_threshold=10)
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    xlbls = ax.get_xmajorticklabels()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    num=-1
    for lbl in xlbls:
        num+=1
        val=colors[dn['leaves'][num]]
        lbl.set_color(val)
       
    if legend:
        for k, each in enumerate(list_of_tasks_complete):
            ax.text(160, 0.8 - (0.8/12) * k, str(k) +'. '+each,
                color = colors[k], fontsize=15)
    plt.savefig(path, format = 'svg')
    
print(len(list_of_tasks_complete))
print(len(list_of_tasks))
viz(gt_affinities, '../exp3_gt.svg', legend = False)

attacks = ['pgd'] #, 'pgd', 'fgsm-l1', 'fgsm-l2', 'pgd-l1', 'pgd-l2']
eps = [0.06]
for attack in attacks:
    for e in eps: 
        with open('../pkl/'+ attack + '_' + str(e)+ '_results.pkl', 'rb') as file:
            results = pickle.load(file)
        tournament_matrices = {}
        tournament_rankings = []
        for target_task in list_of_tasks:
            tournament_matrices[target_task] = np.zeros(shape = (num_task, num_task))  
            for i in range(num_task):
                for j in range(num_task):
                    tournament_matrices[target_task][i][j] = compete(results[list_of_tasks[i]][target_task], results[list_of_tasks[j]][target_task])
            tournament_rankings.append(np.argsort(-np.sum(tournament_matrices[target_task] > tournament_matrices[target_task].T, axis = -1)))
        adv_affinities = np.zeros(shape = (num_task, num_task))
        for i,k in enumerate(list_of_tasks):
            eig_vals,eig_vecs = eigh(tournament_matrices[k]/tournament_matrices[k].T)
            idx = np.argmax(np.abs(eig_vals))
            p_eig_vec = np.exp(-20 * softmax(np.abs(eig_vecs[:, idx])))
            adv_affinities[i] = p_eig_vec
        viz(adv_affinities, '../exp3_prediction.svg', legend = True)
        for t in range(15):
            print('task name:',list_of_tasks[t])
            print('ground truth prediction:',[list_of_tasks[idx] for idx in np.argsort(gt_affinities[t])])
            print('ranking prediction:', [list_of_tasks[idx] for idx in tournament_rankings[t]])
            print('adv eigen value prediction:',[list_of_tasks[idx] for idx in np.argsort(adv_affinities[t])])
            print('=====================')

from sklearn.cluster import AgglomerativeClustering
import scipy
entropy = []
def compute_entropy(cluster):
    total = 0
    d = {}
    for each in cluster:
        if each in d:
            d[each]+=1
        else:
            d[each] =1
        total+=1
    tmp = []
    for k,v in d.items():
        tmp.append(v/total)
    return scipy.stats.entropy(tmp)
    
for n in range(1, len(list_of_tasks) + 1):
    gt_clustering = AgglomerativeClustering(n_clusters = n).fit(gt_affinities.T)
    gt = gt_clustering.labels_
    adv_clustering = AgglomerativeClustering(n_clusters = n).fit(adv_affinities.T)
    adv = adv_clustering.labels_
    e = 0
    for i in range(n):
        tmp = []
        for j,each in enumerate(adv):
            if each == i:
                tmp.append(gt[j])
        e+=compute_entropy(tmp)
    entropy.append(e/n)
    
sns.set(style="darkgrid")
plt.plot(np.arange(1, len(entropy) + 1), entropy)
plt.title('Average inner cluster entropy with different granularity')
plt.ylabel('entropy')
plt.xlabel('number of clusters')
plt.savefig('../exp3_entropy.svg', format = 'svg')




