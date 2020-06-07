import numpy as np
import pickle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.linalg import eigh
import seaborn as sns

def viz(scores, similarity, list_of_attributes, list_of_datasets, top_k = 10):
    idx = np.argsort(-similarity)[:top_k]
    scores = scores[idx+1]
    similarity = similarity[idx]
    list_of_attributes = [list_of_attributes[i] for i in idx]
    barWidth = 0.1
    plt.figure(figsize = (16,8))
    pal = sns.color_palette(n_colors = len(list_of_datasets) + 1)
    for i in range(7):
        bar = scores[:,i]
        r = [each + barWidth * i for each in np.arange(len(bar))]
        plt.bar(r, bar, color= pal[i], width=barWidth, edgecolor='white', label=list_of_datasets[i])
    
    bar = similarity
    r = [each + barWidth * 7 for each in np.arange(len(bar))]
    plt.bar(r, bar, color= pal[-1], width=barWidth, edgecolor='white', label='Adv transferability')
    
    plt.yticks(fontsize=20)
    plt.xticks([r + 3 * barWidth for r in range(len(bar))], list_of_attributes, rotation=0,fontsize=20)
    plt.legend(fancybox=True, framealpha=0.5, ncol=2, prop={'size': 20})


# In[2]:


list_of_datasets = " LFW CFP-FF CFP-FP AgeDB CALFW CPLFW VGG2-FP"
list_of_datasets = list_of_datasets.split()


# In[3]:


list_of_attributes = ['Shadow',
 'Arched Eyebrows',
 'Attractive',
 'Bags under Eyes',
 'Bald',
 'Bangs',
 'Big Lips',
 'Big Nose',
 'Black Hair',
 'Blond Hair',
 'Blurry',
 'Brown Hair',
 'Bushy Eyebrows',
 'Chubby',
 'Double Chin',
 'Eyeglasses',
 'Goatee',
 'Gray Hair',
 'Heavy Makeup',
 'High Cheekbones',
 'Male',
 'Mouth Slightly Open',
 'Mustache',
 'Narrow Eyes',
 'No Beard',
 'Oval Face',
 'Pale Skin',
 'Pointy Nose',
 'Receding Hairline',
 'Rosy Cheeks',
 'Sideburns',
 'Smiling',
 'Straight Hair',
 'Wavy Hair',
 'Wearing Earrings',
 'Wearing Hat',
 'Wearing Lipstick',
 'Wearing Necklace',
 'Wearing Necktie',
 'Young']


# In[4]:


scores = np.array([[0.774,0.768,0.604,0.555,0.627,0.584,0.608],
       [0.611,0.650,0.573,0.520,0.516,0.525,0.540],
       [0.576,0.640,0.514,0.553,0.556,0.500,0.522],
       [0.608,0.622,0.561,0.554,0.518,0.540,0.536],
       [0.560,0.573,0.535,0.497,0.509,0.522,0.542],
       [0.620,0.632,0.576,0.501,0.538,0.529,0.548],
       [0.608,0.578,0.543,0.533,0.532,0.527,0.548],
       [0.599,0.621,0.543,0.533,0.537,0.519,0.543],
       [0.595,0.625,0.540,0.522,0.526,0.513,0.530],
       [0.607,0.656,0.557,0.561,0.528,0.518,0.536],
       [0.640,0.663,0.540,0.509,0.539,0.523,0.547],
       [0.553,0.583,0.540,0.504,0.522,0.517,0.510],
       [0.598,0.619,0.532,0.537,0.525,0.523,0.539],
       [0.583,0.525,0.540,0.514,0.534,0.521,0.543],
       [0.609,0.628,0.526,0.500,0.527,0.516,0.522],
       [0.602,0.595,0.533,0.501,0.527,0.524,0.545],
       [0.658,0.642,0.592,0.505,0.565,0.558,0.567],
       [0.596,0.632,0.596,0.495,0.539,0.544,0.555],
       [0.611,0.614,0.557,0.499,0.522,0.540,0.535],
       [0.593,0.612,0.572,0.545,0.529,0.520,0.562],
       [0.533,0.559,0.518,0.500,0.504,0.515,0.523],
       [0.615,0.663,0.616,0.608,0.530,0.520,0.577],
       [0.539,0.559,0.510,0.500,0.519,0.508,0.507],
       [0.665,0.674,0.617,0.535,0.550,0.542,0.566],
       [0.556,0.529,0.517,0.507,0.511,0.510,0.526],
       [0.572,0.588,0.579,0.510,0.516,0.522,0.541],
       [0.619,0.654,0.531,0.497,0.522,0.501,0.523],
       [0.585,0.565,0.538,0.506,0.511,0.532,0.552],
       [0.595,0.612,0.520,0.498,0.536,0.509,0.531],
       [0.610,0.624,0.513,0.510,0.543,0.536,0.529],
       [0.537,0.564,0.525,0.521,0.490,0.505,0.515],
       [0.606,0.620,0.568,0.495,0.517,0.518,0.542],
       [0.526,0.554,0.503,0.497,0.519,0.510,0.531],
       [0.571,0.590,0.536,0.537,0.526,0.526,0.531],
       [0.586,0.624,0.564,0.523,0.524,0.525,0.560],
       [0.576,0.607,0.537,0.499,0.515,0.514,0.536],
       [0.603,0.687,0.580,0.535,0.545,0.540,0.549],
       [0.589,0.610,0.557,0.545,0.511,0.524,0.546],
       [0.661,0.663,0.582,0.509,0.546,0.545,0.553],
       [0.598,0.581,0.530,0.506,0.514,0.513,0.526],
       [0.667,0.689,0.623,0.500,0.529,0.565,0.579]
      ])


# In[5]:


def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def compete(loss1, loss2):
    count = 0
    total = len(loss1)
    for i in range(total):
        if loss1[i]>loss2[i]:
            count+=1
    return np.clip(count/total, 0.001, 0.999)

#attacks = ['fgsm', 'pgd', 'fgsm-l1', 'fgsm-l2', 'pgd-l1', 'pgd-l2']
#eps = [0.03,0.06, 0.1]
attacks = ['pgd']
eps = [0.03]
for attack in attacks:
    for e in eps:
        with open(attack + '_' + str(e) + '_results.pkl','rb') as file:
            results = pickle.load(file)
        num = len(list(results.keys()))
        average_loss = [np.mean(results[str(idx)]) for idx in range(40)]
        p_eig_vec = np.array(average_loss)/np.max(average_loss)
        viz(scores,p_eig_vec,list_of_attributes, list_of_datasets, top_k = 5)
        plt.savefig('../exp2_result.svg', format='svg')
