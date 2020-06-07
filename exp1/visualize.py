import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def softmax(x):
    return np.exp(20 * x)/sum(np.exp(20 * x))

transfer_acc = []
transfer_last_acc = []
for i in range(5):
    transfer_acc.append(np.load('npy/transfer_test_acc_'+str(i*25)+'_percent.npy'))
    transfer_last_acc.append(np.load('npy/transfer_last_test_acc_'+str(i*25)+'_percent.npy'))
def compete(loss1, loss2):
    count = 0
    total = len(loss1)
    for i in range(total):
        if loss1[i]>loss2[i]:
            count+=1
    return np.clip(count/total, 0.001, 0.999)

#attacks = ['fgsm', 'pgd', 'fgsm-l1', 'fgsm-l2', 'pgd-l1', 'pgd-l2']
#eps = [0.03, 0.06, 0.1, 0.2]
attacks = ['pgd']
eps = [0.06]
for attack_method in attacks:
    for e in eps:
        print( attack_method + '_' +str(e))
        with open( attack_method + '_' +str(e) + '_per_img_results.pkl','rb') as file:
            results = pickle.load(file)

        labels = ['100%','75%', '50%', '25%', '0%']

        num = len(list(results.keys()))
        p_eig_vec = np.array([np.mean(results[str(i)]) for i in range(5)])
        task_transfer_acc_last = [each[-1] for each in transfer_last_acc]
        task_transfer_acc = [each[-1] for each in transfer_acc]
        x = np.arange(len(labels))
        width = 0.2

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, p_eig_vec, width, label='adv transferabiliy')
        rects2 = ax.bar(x, np.array(task_transfer_acc_last)/100, width, label='transfer acc')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('% of animals', fontsize=12)
        ax.legend(prop={'size': 12})
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig('exp1_result.svg')

sns.set(style="whitegrid")
transfer_loss = []
transfer_last_loss = []
transfer_acc = []
transfer_last_acc = []
for i in range(5):
    transfer_loss.append(np.load('npy/transfer_test_loss_'+str(i*25)+'_percent.npy'))
    transfer_acc.append(np.load('npy/transfer_test_acc_'+str(i*25)+'_percent.npy'))
    transfer_last_loss.append(np.load('npy/transfer_last_test_loss_'+str(i*25)+'_percent.npy'))
    transfer_last_acc.append(np.load('npy/transfer_last_test_acc_'+str(i*25)+'_percent.npy'))


labels = ['100%','75%', '50%', '25%', '0%']
task_transfer_acc_last = [each[-1] for each in transfer_last_acc]
task_transfer_acc = [each[-1] for each in transfer_acc] 

fig, ax = plt.subplots()
x = np.arange(transfer_loss[0].shape[0]) 
line_style = [':',':','-.','--','-']
for i in range(5):
    ax.plot(x, transfer_acc[i]/100, label = labels[i] + ' animals,' + str( i * 25) + '% vehicles ', linestyle = line_style[i])

ax.set_ylabel('Acc', fontsize = 12.5)
ax.set_xlabel('Epoch', fontsize = 12.5)
ax.legend(prop={'size': 12.5})
plt.xticks(fontsize=12.5)
plt.yticks(fontsize=12.5)
plt.savefig('exp1_fine_tune.svg')
