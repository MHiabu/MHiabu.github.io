
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(OpenML)
library(mlr3pipelines)
library(future)
library(tidyverse) # For easy data manipulation
library(rpart)
library(rpart.plot)
future::plan("multisession")

# load credit-g data and define task
credit_data = getOMLDataSet(data.id = 31)
task = as_task_classif(credit_data$data, target = "class")

# a-----------------------------------------------------------------------------

# set up a tree model
full_tree <- lrn("classif.rpart", predict_type="prob")
full_tree$train(task)

# load credit-g data and define task
#full_tree_trained <- full_tree$model$classif.rpart$model # Only needed if we have preprocessing layers
full_tree_trained <- full_tree$model
plot(full_tree_trained , compress = TRUE, margin = 0.1)
text(full_tree_trained , use.n = TRUE, cex=0.8)

rpart.plot(full_tree_trained, cex=0.1, extra=4) # requires the rpart.plot package

# b ----------------------------------------------------------------------------
my_cart_learner_cv = lrn("classif.rpart", xval = 5, predict_type = "prob")
my_cart_learner_cv$train(task)
cart_trained_cv = my_cart_learner_cv$model
rpart::plotcp(cart_trained_cv)
rpart::printcp(cart_trained_cv)

# c ----------------------------------------------------------------------------
my_pruned_tree_model = lrn("classif.rpart", cp=0.017, predict_type="prob")
my_pruned_tree_model$train(task)

rpart.plot(my_pruned_tree_model$model, extra=4) # requires the rpart.plot package

# Two example predictions - One good and one bad
# Just in order to check how the plots should be read
pred = my_pruned_tree_model$predict(task)
cbind(task$data(), pred$response) %>% head(2)

# d ----------------------------------------------------------------------------

baseline = lrn("classif.featureless", id="featureless", predict_type="prob")
cart = lrn("classif.rpart", id="cart", cp=0, predict_type="prob")
cart_pruned = lrn("classif.rpart", id="cart_pruned", cp=0.017, predict_type="prob")
xgb = lrn(
  "classif.xgboost",
  eta=to_tune(0, 0.5),
  nrounds=to_tune(10,  5000),
  max_depth=to_tune(1, 10),
  predict_type="prob",
  id="xgb"
  ) %>% 
  auto_tuner(
    tuner=tnr("random_search"),
    learner=.,
    resampling=rsmp("cv", folds=3),
    terminator = trm("evals", n_evals=30)
  )
xgb = (
  po("encode", method="treatment", affect_columns=selector_type("factor")) %>>%
    xgb
) %>% as_learner()
rf = lrn(
  "classif.ranger",
  mtry.ratio=to_tune(0.1, 1),
  min.node.size=to_tune(1, 50),
  predict_type="prob",
  id="rf"
  ) %>% 
  auto_tuner(
    tuner=tnr("random_search"),
    learner=.,
    resampling=rsmp("cv", folds=3),
    terminator = trm("evals", n_evals=30)
  )

bm_grid = benchmark_grid(
  tasks=task,
  learners = list(baseline, cart, cart_pruned, xgb, rf),
  resamplings = rsmp("cv", folds=4)
)

res = benchmark(bm_grid)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

# e ----------------------------------------------------------------------------
# Create a new measure
costs = matrix(c(0, 5, 1, 0), nrow = 2)
dimnames(costs) = list(truth = task$class_names, predicted = task$class_names)
print(costs)
#>       predicted
#> truth  good bad
#>   good    0   1
#>   bad     5   0
# mlr3 needs truth in columns, predictions in rows
costs = t(costs)
new_cost_measure = msr("classif.costs", costs = costs)
res$aggregate(new_cost_measure)

# f ----------------------------------------------------------------------------

baseline_costs = lrn("classif.featureless", id="featureless_costs", predict_type="prob")
cart_costs = lrn(
  "classif.rpart",
  id="cart_costs",
  cp=0,
  predict_type="prob"
  )
cart_pruned_costs = lrn(
  "classif.rpart",
  id="cart_pruned_costs",
  cp=0.017,
  predict_type="prob"
  )
xgb_costs = lrn(
  "classif.xgboost",
  eta=to_tune(0, 0.5),
  nrounds=to_tune(10,  5000),
  max_depth=to_tune(1, 10),
  predict_type="prob",
  id="xgb_costs"
) %>% 
  auto_tuner(
    tuner=tnr("random_search"),
    learner=.,
    resampling=rsmp("cv", folds=3),
    terminator = trm("evals", n_evals=30),
    measure = msr("classif.costs", costs = costs)
  )
xgb_costs = (
  po("encode", method="treatment", affect_columns=selector_type("factor")) %>>%
    xgb_costs
) %>% as_learner()
rf_costs = lrn(
  "classif.ranger",
  mtry.ratio=to_tune(0.1, 1),
  min.node.size=to_tune(1, 50),
  predict_type="prob",
  id="rf_costs"
) %>% 
  auto_tuner(
    tuner=tnr("random_search"),
    learner=.,
    resampling=rsmp("cv", folds=3),
    measure = msr("classif.costs", costs = costs),
    terminator = trm("evals", n_evals=30)
  )

bm_grid_costs = benchmark_grid(
  tasks=task,
  learners = list(baseline_costs, cart_costs, cart_pruned_costs, xgb_costs, rf_costs),
  resamplings = rsmp("cv", folds=4)
)

res_costs = benchmark(bm_grid_costs)

res_costs$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr"),
                   msr("classif.costs", costs = costs)))

