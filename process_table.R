library(tidyverse)
library(ggplot2)

setwd('/Users/chloekwon/Desktop/projects/politeness-agreement')

sent_df <- read.csv('data/sentences.csv')
res_df <- read.csv('data/results_evaluation_v3.csv')

colnames(sent_df)
colnames(res_df)

sent_df <- sent_df %>%
  select(item_number, sentence, distance, subject, verb_phrase, grammatical)

df <- merge(res_df, sent_df, by="sentence")
names(df)

write.csv(df, 'data/results_evaluation.csv')
