library(tidymodels)
library(ggplot2)
library(visdat)
library(corrplot)
library(janitor)


data = read.csv('wine.csv')

data = data %>% clean_names()

#eda

##NA
vis_dat(data)

#Delete
data = data[,-'fixed.acidity']


##split
data_split <- data %>% 
        initial_split(prop = 0.7)

train <- training(data_split)
test <- testing(data_split)

##corrplot
M = cor(train)
corrplot(M, method = 'color', col = COL2(n=20), cl.length = 21, order = 'AOE',
         addCoef.col = 'grey')

##plot
data %>% ggplot(aes(x=quality)) +
        geom_bar()


train %>% ggplot(aes(x = as.factor(quality), y = alcohol)) +
        geom_boxplot()

train %>% ggplot(aes(x = as.factor(quality), y = volatile_acidity)) +
        geom_boxplot()

folds <- vfold_cv(train, v = 5)

recipe = recipe(quality ~ ., data = train) %>%
        step_normalize(all_predictors())


tree_model = decision_tree(cost_complexity = tune()) %>%
        set_engine('rpart') %>%
        set_mode('regression')

tree_wf = workflow() %>%
        add_model(tree_model) %>%
        add_recipe(recipe)

tree_grid <- grid_regular(cost_complexity(range = c(-3, -1)), 
                          levels = 2)

tree_tune <- tree_wf %>% 
        tune_grid(resamples = folds, 
                  grid = tree_grid)

save(tree_tune, tree_wf, file = "tree_tune.rda")

autoplot(tree_tune)



rf_model = rand_forest(min_n = tune(),
                       mtry = tune(),
                       mode = "regression") %>% 
        set_engine("ranger")

rf_wf = workflow() %>% 
        add_model(rf_model) %>% 
        add_recipe(recipe)

rf_params = parameters(rf_model) %>% 
        update(mtry = mtry(range= c(2, 120)))

rf_grid = grid_regular(rf_params, 
                       levels = 2)

rf_tune = rf_wf %>% 
        tune_grid(resamples = folds, 
                  grid = rf_grid,
                  metrics = metric_set(rmse, rsq))

save(rf_tune, rf_wf, file = "rf_tune.rda")

autoplot(rf_tune)



bt_model = boost_tree(min_n = tune(),
                      mtry = tune(),
                      learn_rate = tune(),
                      mode = "regression") %>% 
        set_engine("xgboost")

bt_wf = workflow() %>% 
        add_model(bt_model) %>% 
        add_recipe(recipe)

bt_params = parameters(bt_model) %>% 
        update(mtry = mtry(range= c(2, 120)),
               learn_rate = learn_rate(range = c(-5, 0.2)))

bt_grid = grid_regular(bt_params, levels = 2)

bt_tune = bt_wf %>% 
        tune_grid(resamples = folds, 
                  grid = bt_grid,
                  metrics = metric_set(rmse, rsq))

save(bt_tune, bt_wf, file = "bt_tune.rda")

autoplot(bt_tune)




