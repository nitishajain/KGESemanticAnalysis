import pandas as pd
import numpy as np
import itertools
import collections
pd.set_option('display.max_columns', None)
import os
from string import digits
import matplotlib.pyplot as plt
import seaborn as sns


entity_types_df = pd.read_csv("SimpleTypeFactsWordnetLevel.tsv", sep="\t", names=['entity', 'type', 'class'], header=None)
print(entity_types_df.head(10))

type_freq = entity_types_df.groupby(['class']).size().sort_values(ascending=False).reset_index(name='count')
total = type_freq['count'].sum()
print(total)


type_freq_abovek = type_freq[type_freq['count'] > 1000]
type_freq_abovek[['class']] = type_freq_abovek[['class']].replace({'<':''}, regex=True)
type_freq_abovek[['class']] = type_freq_abovek[['class']].replace({'>':''}, regex=True)
type_freq_abovek[['class']] = type_freq_abovek[['class']].replace({'wordnet':''}, regex=True)
type_freq_abovek[['class']] = type_freq_abovek[['class']].replace({'_':''}, regex=True)
type_freq_abovek[['class']] = type_freq_abovek[['class']].replace({'\d+':''},regex=True)


plt.rcParams.update({'font.size': 15})
dataset = type_freq_abovek

print(dataset.head())
#ax = sns.distplot(dataset['count'])
#ax.set_xscale('log')

#ax = sns.stripplot(x='class', y='count', data=dataset, jitter=False)
ax = sns.barplot(x='class', y='count', data=dataset)
print("Plotting")
plt.xticks(rotation=90)

plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.25)
plt.gcf().subplots_adjust(left=0.06)
#plt.gcf().subplots_adjust(right=-0.06)
ax.set_ylabel('Frequency of entities')
ax.set_xlabel('')

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.show()
