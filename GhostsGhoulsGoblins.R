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
install.packages('vroom')
library(vroom)
install.packages('parsnip')
library(parsnip)
install.packages('dbarts')
library(dbarts)
install.packages('embed')
library(embed)
library(themis)

# rm(list=ls()) use to erase environment

## 112 Cols

data_train <- vroom("./data/train.csv") %>%
  mutate(type=factor(type))# grab training data
# Missing values fill experiment:
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


###############################
##### K-Nearest Neighbors #####
###############################

## For target encoding ###
my_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  #step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_glm(color, outcome = vars(type)) %>% 
  #step_pca(all_predictors(), threshold = 0.8) # Threshold between 0 and 1, test run for classification rf
  #step_smote(all_outcomes(), neighbors = 5)
  step_dummy(all_nominal_predictors()) # get dummy variables

#install.packages('kknn')
library(kknn)

my_recipe_k <- my_recipe %>%
  step_normalize()

prepped_recipe_k <- prep(my_recipe) # preprocessing new data
baked_data_k <- bake(prepped_recipe_k, new_data = data_train)

## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe_k) %>%
  add_model(knn_model)

## Fit or Tune MOdel
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(data_train, v = 10, repeats = 1)

# Run CV
CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best('accuracy')

final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = data_train)

data_test <- vroom("./data/test.csv") # grab testing data

# Kaggle DF
ggg_predictions <- predict(final_wf,
                              new_data=data_test,
                              type="class") %>% # "class" or "prob"
  mutate(id = data_test$id, type = .pred_class) %>%
  select(id, type)

vroom_write(ggg_predictions, "./data/ggg_pred_knn.csv", delim = ",")
# save(file = 'amazon_knn_wf.RData', list = c('final_wf'))
# load('amazon_knn_wf.RData')


###########################
##### Neural Networks #####
###########################

nn_recipe <- recipe(rFormula, data= data_train) %>%
  update_role(id, new_role="id") %>%
  step_lencode_glm(color, outcome = vars(type)) %>% ## Turn color to factor then dummy encode color
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 50, #or 100 or 250
                activation="relu") %>% # RELU works comparatively well
  set_engine("keras", verbose=0) %>% #verbose = 0 prints off less
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 10)),
                            levels=5)

# Split data for CV
folds <- vfold_cv(data_train, v = 10, repeats = 1)

# Run CV
CV_results <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuneGrid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best('accuracy')

final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = data_train)

# MLP in terms of accuracy
tuned_nn %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results22
## This takes a few min (10 on my laptop) so run it on becker if you want











