import pandas as pd
import lifelines
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test,multivariate_logrank_test

from sklearn.metrics import accuracy_score, roc_auc_score

import os

import numpy as np


# df = pd.read_csv("clinSurv_gpSlide_data.csv")
df = pd.read_csv("clinic_ki_desmo.csv")
df.rename(columns={'Unnamed: 0':'id'}, inplace=True)
df = df.fillna(0)

# df["growth_pattern"] = ""
# df.loc[df.percent_desmoplastic >= 0.5, 'growth_pattern'] = 'des'
# df.loc[df.percent_pushing < 0.5 , 'growth_pattern'] = 'nd'
# df.loc[df.percent_replacement > 0.5, 'growth pattern'] = 'rep'
# df.loc[(df.percent_replacement < 0.5) & (df.percent_pushing > 0.5) & (df.percent_replacement > 0.5), 'growth pattern'] = 'mixed'
#print(df.head(10))

# print(len(df))


df_p=pd.read_csv('./output/intermediate/ki_invasion_imagenet_celltype_relation.csv')
df_p.rename(columns={'slide_name':'slide'}, inplace=True)
df_p.rename(columns={'patient_id':'id'}, inplace=True)
# df_clinical = df.groupby('id').mean()
# df = df.groupby('id').agg('mean')


#print(df_clinical.head())
#print(df.head(5))
# print(len(df))

#print(df_clinical.head())

df_ip=pd.read_csv('result_Tumor_tils_density_test.csv')
df_ip.rename(columns={'file_name':'slide'}, inplace=True)





df_ip=pd.merge(df_ip, df, on='id',how='inner')

print(df_ip.columns)

#print(df_ip.head(5))

#print(df_ip.head(30)) 
# df_ip["immune phenotype"] = ""
#df_ip["Mucinous_Status"] = ""
# df_ip.loc[df_ip.immune_infl >= 0.1, 'immune phenotype'] = 'IN'
# df_ip.loc[(df_ip.immune_infl < 0.1) & (df_ip.immune_excl >= 0.1), 'immune phenotype'] = 'IE'
# df_ip.loc[(df_ip.immune_infl < 0.1) & (df_ip.immune_excl < 0.1), 'immune phenotype'] = 'ID'
# df_ip.loc[df_ip.immune_infl > 0.1, 'Mucinous_Status'] = 'muc'
# df_ip.loc[(df_ip.immune_infl <= 0.1) & (df_ip.immune_excl > 0.1), 'Mucinous_Status'] = 'nm'
# df_ip.loc[(df_ip.immune_infl <= 0.1) & (df_ip.immune_excl <= 0.1), 'Mucinous_Status'] = 'nm'
# df_ip.loc[df_ip.immune_excl > 0.1, 'immune phenotype'] = 'IE'
# df_ip.loc[(df_ip.immune_excl <= 0.1) & (df_ip.immune_infl > 0.1), 'immune phenotype'] = 'IN'
# df_ip.loc[(df_ip.immune_excl <= 0.1) & (df_ip.immune_infl <= 0.1), 'immune phenotype'] = 'ID'

# df_ip["desmoplastic_mucinous_PFS"] = ""
# df_ip.loc[(df_ip.percent_desmoplastic >= 50) & (df_ip.immune_infl > 0.1), 'desmoplastic_mucinous_PFS'] = 'des_muc'
# df_ip.loc[(df_ip.percent_desmoplastic < 50) & (df_ip.immune_infl > 0.1), 'desmoplastic_mucinous_PFS'] = 'nondes_muc'
# df_ip.loc[(df_ip.percent_desmoplastic >= 50) & (df_ip.immune_infl <= 0.1), 'desmoplastic_mucinous_PFS'] = 'des_nonmuc'
# df_ip.loc[(df_ip.percent_desmoplastic < 50) & (df_ip.immune_infl <= 0.1), 'desmoplastic_mucinous_PFS'] = 'nondes_nonmuc'

# df_ip["desmoplastic_PFS"] = ""
# df_ip.loc[(df_ip.percent_desmoplastic >= 50), 'desmoplastic_PFS'] = 'des'
# df_ip.loc[(df_ip.percent_desmoplastic < 50), 'desmoplastic_PFS'] = 'nondes'

# df_ip["mucinus_PFS"] = ""
# df_ip.loc[(df_ip.immune_infl >= 0.1), 'mucinus_PFS'] = 'muc'
# df_ip.loc[(df_ip.immune_infl < 0.1), 'mucinus_PFS'] = 'nonmuc'


# df["desmoplastic_mucinous_OS"] = ""
# df.loc[(df.desmo_status >= 1) & (df.immune_infl > 0.1), 'desmoplastic_mucinous_OS'] = 'des_muc'
# df.loc[(df.desmo_status < 1) & (df.immune_infl > 0.1), 'desmoplastic_mucinous_OS'] = 'nondes_muc'
# df.loc[(df.desmo_status >= 1) & (df.immune_infl <= 0.1), 'desmoplastic_mucinous_OS'] = 'des_nonmuc'
# df.loc[(df.desmo_status < 1) & (df.immune_infl <= 0.1), 'desmoplastic_mucinous_OS'] = 'nondes_nonmuc'

# df["desmoplastic_OS"] = ""
# df.loc[(df.desmo_status >= 1), 'desmoplastic_OS'] = 'des'
# df.loc[(df.desmo_status < 1), 'desmoplastic_OS'] = 'nondes'

# df["mucinus_OS"] = ""
# df.loc[(df.immune_infl >= 0.1), 'mucinus_OS'] = 'muc'
# df.loc[(df.immune_infl < 0.1), 'mucinus_OS'] = 'nonmuc'

df_ip.loc[df_ip.immune_infl_x < 0.1, 'desmo_score'] = 0#'non-desmoplastic'
df_ip.loc[df_ip.immune_infl_x >= 0.1, 'desmo_score'] = 1#'desmoplastic'



# df_ip.loc[df_ip.percent_desmoplastic >= 50, 'growth_pattern_2'] = 'des'
# df_ip.loc[df_ip.percent_desmoplastic < 50 , 'growth_pattern_2'] = 'non-des'
#df_ip.loc[df_ip.percent_replacement >= 50 , 'growth_pattern_2'] = 'rep'

df_ip = df_ip.drop(["immune_infl_x"], axis=1)
df_ip = df_ip.drop(["immune_excl"], axis=1)
df_ip = df_ip.drop(["immune_des"], axis=1)
# df_ip = df_ip.drop(["percent_desmoplastic"], axis=1)
# df_ip = df_ip.drop(["percent_pushing"], axis=1)
# df_ip = df_ip.drop(["percent_replacement"], axis=1)

df_ip = df_ip.groupby('id').agg('mean')
print(df_ip.head(5))
df_ip.to_csv('desmo_pred.csv')
df_ip.loc[df_ip.desmo_score >= 1, 'HGP_pred'] = 'desmoplastic'
df_ip.loc[df_ip.desmo_score < 1, 'HGP_pred'] = 'non-desmoplastic'
df_ip.loc[df_ip.desmo_status >= 1, 'HGP_label'] = 'desmoplastic'
df_ip.loc[df_ip.desmo_status < 1, 'HGP_label'] = 'non-desmoplastic'
acc = accuracy_score(df_ip.HGP_label, df_ip.HGP_pred)
# auc = roc_auc_score(df_ip.desmo_status, df_ip.desmo_score)
print('acc=',acc)

df_ip.to_csv('desmo_pred.csv')
#df_ip2 = df_

# print(df_ip.head(5))


#s=df['id'].values
# miss=[]

# for i in df_p['id']:
#     if i not in s:
#         miss.append(i)
# print(len(miss))
# print(miss)

# df_m=pd.merge(df_p,df_clinical[['OS_in_days','alive_0_yes_1_no']], on='id',how='inner')
# df_m=pd.merge(df_m, df_ip, on='slide',how='inner')

# df_m=pd.merge(df, df_ip, on='slide',how='inner')
df_m=df_ip

print(df_m.columns)

df_m['OS_in_days'] = df_m['OS_in_days']/365
df_m['PFS_in_days'] = df_m['PFS_in_days']/365

# df_m2 = df_m.query("desmoplastic_mucinous_OS=='nondes_nonmuc' or desmoplastic_mucinous_OS=='nondes_muc'")
# df_m3 = df_m.query("desmoplastic_mucinous_OS=='nondes_nonmuc' or desmoplastic_mucinous_OS=='des_muc'")
# df_m4 = df_m.query("desmoplastic_mucinous_OS=='nondes_nonmuc' or desmoplastic_mucinous_OS=='des_nonmuc'")
# df_m5 = df_m.query("desmoplastic_mucinous_OS=='des_muc' or desmoplastic_mucinous_OS=='des_nonmuc'")


# print(df_m.columns)
print(len(df_m))
print(df_m.head(5))

#df_m = df_m.drop(['slide'], axis=1)

#print(df_m.head(5))

#df_m = df_m.drop(["t-h same tile rate"], axis=1)
#df_m = df_m.drop(["t-f same tile rate"], axis=1)
#df_m = df_m.drop(["t-i same tile rate"], axis=1)
#df_m = df_m.drop(["t-b same tile rate"], axis=1)
#df_m = df_m.drop(["t-bd same tile rate"], axis=1)
#df_m = df_m.drop(["t-mf same tile rate"], axis=1)
#df_m = df_m.drop(["t-n same tile rate"], axis=1)
#df_m = df_m.drop(["i-f same tile rate"], axis=1)
#df_m = df_m.drop(["h-n same tile rate"], axis=1)

# df_m_mean = df_m.groupby('id').agg('mean')
# df_m_median = df_m.groupby('id').agg('median')
# idx = df_m.groupby(['id'])['tumor'].transform(max) == df_m['tumor']
# df_m_max = df_m[idx]
# df_m_sum = df_m.groupby('id').agg('mean')

    
# df_m_sum = df_m_sum.drop(["t-h same"], axis=1)
# df_m_sum = df_m_sum.drop(["t-h touch"], axis=1)
# df_m_sum = df_m_sum.drop(["t-f same"], axis=1)
# df_m_sum = df_m_sum.drop(["t-f touch"], axis=1)
# df_m_sum = df_m_sum.drop(["t-i same"], axis=1)
# df_m_sum = df_m_sum.drop(["t-i touch"], axis=1)
# df_m_sum = df_m_sum.drop(["t-b same"], axis=1)
# df_m_sum = df_m_sum.drop(["t-b touch"], axis=1)
# df_m_sum = df_m_sum.drop(["t-bd same"], axis=1)
# df_m_sum = df_m_sum.drop(["t-bd touch"], axis=1)
# df_m_sum = df_m_sum.drop(["t-mf same"], axis=1)
# df_m_sum = df_m_sum.drop(["t-mf touch"], axis=1)
# df_m_sum = df_m_sum.drop(["t-n same"], axis=1)
# df_m_sum = df_m_sum.drop(["t-n touch"], axis=1)
# df_m_sum = df_m_sum.drop(["t-m same"], axis=1)
# df_m_sum = df_m_sum.drop(["t-m touch"], axis=1)
# df_m_sum = df_m_sum.drop(["i-f same"], axis=1)
# df_m_sum = df_m_sum.drop(["i-f touch"], axis=1)
# df_m_sum = df_m_sum.drop(["h-n same"], axis=1)
# df_m_sum = df_m_sum.drop(["h-n touch"], axis=1)
# df_m_sum = df_m_sum.drop(["tumor"], axis=1)
# df_m_sum = df_m_sum.drop(["hepatocyte"], axis=1)
# df_m_sum = df_m_sum.drop(["fibrosis"], axis=1)
# df_m_sum = df_m_sum.drop(["inflammation"], axis=1)
# df_m_sum = df_m_sum.drop(["blood"], axis=1)
# df_m_sum = df_m_sum.drop(["bileduct"], axis=1)
# df_m_sum = df_m_sum.drop(["macrophage"], axis=1)
# df_m_sum = df_m_sum.drop(["necrosis"], axis=1)
# df_m_sum = df_m_sum.drop(["mucin"], axis=1)
# df_m_sum["immune phenotype"] = ""
# df_m_sum.loc[df_m_sum.immune_infl >= 0.1, 'immune phenotype'] = 'IN'
# df_m_sum.loc[(df_m_sum.immune_infl < 0.1) & (df_m_sum.immune_excl >= 0.05), 'immune phenotype'] = 'IE'
# df_m_sum.loc[(df_m_sum.immune_infl < 0.1) & (df_m_sum.immune_excl < 0.05), 'immune phenotype'] = 'ID'
# df_m_sum.to_csv("immune_subtype.csv")
# df_m_sum = df_m_sum.drop(["immune_infl"], axis=1)
# df_m_sum = df_m_sum.drop(["immune_excl"], axis=1)
# df_m_sum = df_m_sum.drop(["immune_des"], axis=1)



def split_group(df) :
    print(df.columns)
    for j in df.columns[:-3]:
        per=df[j].quantile(0.5)
        print(j, per)

        for index, row in df.iterrows() :
            if float(row[j])<=float(per):
                df[j][index]='low'
            elif float(row[j])>float(per):
                df[j][index]='high'
            # if float(row[j])<0.5:
            #     df[j][index]='low'
            # elif float(row[j])>=0.5:
            #     df[j][index]='high'
            
    # df_ip.loc[df_ip.immune_infl >= 0.1, 'immune phenotype'] = 'IN'
    # df_ip.loc[(df_ip.immune_infl < 0.1) & (df_ip.immune_excl >= 0.1), 'immune phenotype'] = 'IE'
    # df_ip.loc[(df_ip.immune_infl < 0.1) & (df_ip.immune_excl < 0.1), 'immune phenotype'] = 'ID'
    # df = df.drop(["immune_infl"], axis=1)
    # df = df.drop(["immune_excl"], axis=1)
    # df = df.drop(["immune_des"], axis=1)
                
# print("all!!")
# split_group(df_m)
# print("mean!!")
# split_group(df_m_mean)
# print("median!!")
# split_group(df_m_median)
# print("max!!")
# split_group(df_m_max)
        



# +
# SURVIVAL ANALYSIS PLOTTING FUNCTIONS: copied from github/lifelines and edited for customization
def is_latex_enabled(): 
    '''
    Returns True if LaTeX is enabled in matplotlib's rcParams,
    False otherwise
    '''
    import matplotlib as mpl

    return mpl.rcParams['text.usetex']


def remove_spines(ax, sides):
    '''
    Remove spines of axis.
    Parameters:
      ax: axes to operate on
      sides: list of sides: top, left, bottom, right
    Examples:
    removespines(ax, ['top'])
    removespines(ax, ['top', 'bottom', 'right', 'left'])
    '''
    for side in sides:
        ax.spines[side].set_visible(False)
    return ax


def move_spines(ax, sides, dists):
    '''
    Move the entire spine relative to the figure.
    Parameters:
      ax: axes to operate on
      sides: list of sides to move. Sides: top, left, bottom, right
      dists: list of float distances to move. Should match sides in length.
    Example:
    move_spines(ax, sides=['left', 'bottom'], dists=[-0.02, 0.1])
    '''
    for side, dist in zip(sides, dists):
        ax.spines[side].set_position(('axes', dist))
    return ax


def remove_ticks(ax, x=False, y=False):
    '''
    Remove ticks from axis.
    Parameters:
      ax: axes to work on
      x: if True, remove xticks. Default False.
      y: if True, remove yticks. Default False.
    Examples:
    removeticks(ax, x=True)
    removeticks(ax, x=True, y=True)
    '''
    if x:
        ax.xaxis.set_ticks_position('none')
    if y:
        ax.yaxis.set_ticks_position('none')
    return ax

def add_at_risk_counts_CUSTOM(*fitters, **kwargs): 
    '''
    Add counts showing how many individuals were at risk at each time point in
    survival/hazard plots.
    Arguments:
      One or several fitters, for example KaplanMeierFitter,
      NelsonAalenFitter, etc...
    Keyword arguments (all optional):
      ax: The axes to add the labels to. Default is the current axes.
      fig: The figure of the axes. Default is the current figure.
      labels: The labels to use for the fitters. Default is whatever was
              specified in the fitters' fit-function. Giving 'None' will
              hide fitter labels.
    Returns:
      ax: The axes which was used.
    Examples:
        # First train some fitters and plot them
        fig = plt.figure()
        ax = plt.subplot(111)
        f1 = KaplanMeierFitter()
        f1.fit(data)
        f1.plot(ax=ax)
        f2 = KaplanMeierFitter()
        f2.fit(data)
        f2.plot(ax=ax)
        # There are equivalent
        add_at_risk_counts(f1, f2)
        add_at_risk_counts(f1, f2, ax=ax, fig=fig)
        # This overrides the labels
        add_at_risk_counts(f1, f2, labels=['fitter one', 'fitter two'])
        # This hides the labels
        add_at_risk_counts(f1, f2, labels=None)
    '''

    # Axes and Figure can't be None
    ax = kwargs.get('ax', None)
    if ax is None:
        ax = plt.gca()

    fig = kwargs.get('fig', None)
    if fig is None:
        fig = plt.gcf()

    fontsize = kwargs.get('fontsize', None)
    if fontsize is None:
        fontsize = 15
        
    if 'labels' not in kwargs:
        labels = [f._label for f in fitters]
        #print(labels)
    else:
        # Allow None, in which case no labels should be used
        labels = kwargs['labels']
        if labels is None:
            labels = [None] * len(fitters)
    # Create another axes where we can put size ticks
    ax2 = plt.twiny(ax=ax)
    # Move the ticks below existing axes
    # Appropriate length scaled for 6 inches. Adjust for figure size.
    ax2_ypos = -0.20 * 6.0 / fig.get_figheight()
    move_spines(ax2, ['bottom'], [ax2_ypos])
    # Hide all fluff
    remove_spines(ax2, ['top', 'right', 'bottom', 'left'])
    # Set ticks and labels on bottom
    ax2.xaxis.tick_bottom()
    # Match tick numbers and locations
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    
    # Remove ticks, need to do this AFTER moving the ticks
    remove_ticks(ax2, x=True, y=True)
    # Add population size at times
    ticklabels = []
    for tick in ax2.get_xticks():
        lbl = ""
        for f, l in zip(fitters, labels):
            # First tick is prepended with the label
            #print(f,l)
            if tick == ax2.get_xticks()[0] and l is not None:
                if is_latex_enabled():
                    s = "\n{}\\quad".format(l) + "{}"
                else:
                    s = "\n{}   ".format(l) + "{}"
            else:
                s = "\n{}"
            lbl += s.format(f[f >= tick].shape[0])
        ticklabels.append(lbl.strip())
    # Align labels to the right so numbers can be compared easily
    #print(ticklabels)
    ax2.set_xticklabels(ticklabels, ha='right', fontsize=10)

    # Add a descriptive headline.
    ax2.xaxis.set_label_coords(0, ax2_ypos)
    ax2.set_xlabel('At risk', fontsize=10)

    plt.tight_layout()
    return ax2


# -

def plot_km(col,df,T,C):
    #plt.figure()
    kmf = KaplanMeierFitter()
    ax = plt.subplot(111)
     # Create another axes where we can put size ticks
    fitters=[]
    labels=[]
    df_d=pd.DataFrame()
    df_E=pd.DataFrame()
    G=[]
    
    j=0

    for r in df[col].unique():
        
        ix = df[col] == r
        try :
            c=kmf.fit(durations=T[ix], event_observed=C[ix],label=r)
            fitters.append(c.durations)
            df_d=pd.concat([df_d.reset_index(drop=True),T[ix].reset_index(drop=True)],axis=0)
            df_E=pd.concat([df_E.reset_index(drop=True),C[ix].reset_index(drop=True)],axis=0)
            G.append(np.zeros(len(T[ix]))+j)

            labels.append(r)
            kmf.plot(ax=ax,ci_show=False, show_censors=True)
            plt.ylim([0,1.2])
            plt.xlim([0, 5])
            plt.ylabel('Survival probability')
            plt.xlabel('Time in years')
            #plt.legend(bbox_to_anchor=(0, 1.5, 1, 0), loc=2, ncol=2, mode="expand", borderaxespad=0)
            # plt.legend(loc="upper left", mode = "expand", ncol = 3)
            plt.legend(loc="upper left", ncol = 3)
            ax.grid(False)
        except :
            print("error",col)
        j+=1
    d_list=df_d.values.tolist()
    e_list=df_E.values.tolist()
    d_list=np.hstack(d_list)
    e_list=np.hstack(e_list)
    G=np.hstack(G)
    return fitters,labels,ax,d_list,e_list,G


# +
#df=processed_df

def draw_figure(df, value, file_prefix) :
    T=df['OS_in_days']
    C=df['alive_0_yes_1_no'].astype(float)

    output_path = './output/intermediate/' + value + '/'
    
    if os.path.exists(output_path) == False :
        os.makedirs(output_path)
        
    # for i in df.columns[:-1]:
    i='HGP_label'
    plt.figure(figsize=(10, 5))
    fitters,labels,ax,d_list,e_list,G=plot_km(i,df,T,C) 
    #print(len(fitters))
    #print(labels)

    df_rank = pd.DataFrame({
        'durations': d_list,
        'events': e_list,
        'groups': G
    })
    result = multivariate_logrank_test(df_rank['durations'], df_rank['groups'], df_rank['events'])
    result.test_statistic
    #print(result.p_value)
    #result.print_summary()

    from lifelines.plotting import add_at_risk_counts
    import seaborn as sns
    sns.despine()
    newxticks = []
    for x in ax.get_xticks():
        if x >= 0:
            newxticks += [x]

    ax.set_xticks(newxticks)
    plt.title(i)
    plt.text(6,.1,'p={}'.format(round(result.p_value,10),'.3f'))
    if len(fitters)==2:
        ax2=add_at_risk_counts_CUSTOM(fitters[0],fitters[1],ax=ax,fontsize=10,labels=labels)
    elif len(fitters)==3:
        ax2=add_at_risk_counts_CUSTOM(fitters[0],fitters[1],fitters[2],ax=ax,fontsize=10,labels=labels)

    plt.savefig(output_path+file_prefix+'_'+i+'_OS.png')
df_m.replace(np.nan, 0)
# df_m_mean.replace(np.nan, 0)
# df_m_median.replace(np.nan, 0)
# df_m_max.replace(np.nan, 0)

draw_figure(df_m, 'pattern', 'all')
# draw_figure(df_m2, 'pattern', 'nondes_muc')
# draw_figure(df_m3, 'pattern', 'des_muc')
# draw_figure(df_m4, 'pattern', 'des_nonmuc')
# draw_figure(df_m5, 'pattern', 'des_mucnonmuc')

# draw_figure(df_m, 'all')
# draw_figure(df_m_mean, 'mean')
# draw_figure(df_m_median, 'median')
# draw_figure(df_m_max, 'max')
# draw_figure(df_m_sum, 'sum')

# # +
# fitters,labels,ax,d_list,e_list,G=plot_km(s[1]) 
# from lifelines.statistics import logrank_test,multivariate_logrank_test
# df_rank = pd.DataFrame({
#    'durations': d_list,
#    'events': e_list,
#    'groups': G
# })
# result = multivariate_logrank_test(df_rank['durations'], df_rank['groups'], df_rank['events'])
# result.test_statistic
# print(result.p_value)
# result.print_summary()

# from lifelines.plotting import add_at_risk_counts
# import seaborn as sns
# sns.despine()
# newxticks = []
# for x in ax.get_xticks():
#     if x >= 0:
#         newxticks += [x]

# ax.set_xticks(newxticks)
# plt.text(6,.1,'p={}'.format(round(result.p_value,10),'.3f'))
# plt.title(s[1])
# ax2=add_at_risk_counts_CUSTOM(fitters[0],fitters[1],fitters[2],ax=ax,fontsize=10,labels=labels)
# plt.savefig('survival_p.png')
# # -


