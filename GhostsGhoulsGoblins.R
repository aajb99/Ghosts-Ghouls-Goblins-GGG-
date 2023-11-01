#install.packages('tidyverse')
library(tidyverse)
#install.packages('tidymodels')
library(tidymodels)
#install.packages('DataExplorer')
#install.packages("poissonreg")
# library(poissonreg)
#install.packages("glmnet")
#library(glmnet)
#library(patchwork)
# install.packages("rpart")
#install.packages('ranger')
#library(ranger)
#install.packages('stacks')
#library(stacks)
#install.packages('vroom')
library(vroom)
#install.packages('parsnip')
#library(parsnip)
# install.packages('dbarts')
# library(dbarts)
#install.packages('embed')
#library(embed)
#library(themis)

# rm(list=ls()) use to erase environment

## 112 Cols

data_train <- vroom("./data/train.csv") %>%
  mutate(type=factor(type))# grab training data
miss_val_data <- vroom("./data/trainWithMissingValues.csv") %>%
  mutate(type=factor(type))# grab training data

#######################
##### Recipe/Bake #####
#######################

rFormula <- type ~ .

## For target encoding/Random Forests: ###
missval_recipe <- recipe(rFormula, data = miss_val_data) %>% # set model formula and dataset
  step_impute_linear(bone_length, impute_with = imp_vars(all_predictors())) %>%
  step_impute_linear(rotting_flesh, impute_with = imp_vars(all_predictors())) %>%
  step_impute_linear(hair_length, impute_with = imp_vars(all_predictors())) %>%
  step_impute_linear(has_soul, impute_with = imp_vars(all_predictors()))
  
prepped_recipe <- prep(missval_recipe) # preprocessing new data
baked_data1 <- bake(prepped_recipe, new_data = miss_val_data)

rmse_vec(data_train[is.na(miss_val_data)], baked_data1[is.na(miss_val_data)])







