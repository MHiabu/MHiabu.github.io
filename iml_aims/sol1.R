library(mlr3)
library(mlr3learners )
library(mlr3tuning)
library(mlr3mbo)
library(glmnet)
library(OpenML)
library(mlr3pipelines)
library(future)

# Parallel computing
future::plan("multisession") 
#future::plan("multicore")  ## doesn't work on Windows

#################################
####### a (Default fitting)

# load credit-g data and define task
credit_data = getOMLDataSet(data.id = 31)

task = as_task_classif(credit_data$data, target = "class")

# Create train and test set
set.seed(2024)
splits = partition(task, stratify=TRUE, ratio=0.8)

credit_data_train = credit_data$data[splits$train,]
credit_data_test = credit_data$data[splits$test, ]

# create train and test tasks
task_train = as_task_classif(credit_data_train, target = "class") 
task_test = as_task_classif(credit_data_test, target = "class") 


#### define learner: logistic regression

my_lr_learner = lrn("classif.log_reg")


#### create a graph that first dummy-encodes factors, then standardizes the variables and afterwards applies logistic regression
graph_learner_lr = as_learner(
  po("encode", method = "treatment",
     affect_columns = selector_type("factor"), id = "binary_enc") %>>% po("scale") %>>% my_lr_learner
)


######## result on test set

graph_learner_lr$train(task_train)
graph_learner_lr$predict(task_test)$score()









#################################
### b(Cross Validation)

#### define learner

### elastic net
my_elasticnet_learner = lrn("classif.glmnet", # logistic regression
                          s= to_tune(0, 1),
                          alpha=to_tune(0, 1))


#### create a graph that first dummy-encodes factors, and afterwards applies elastic net
graph_learner_elastic_net = as_learner(
  po("encode", method = "treatment",
     affect_columns = selector_type("factor"), id = "binary_enc") %>>% po("scale") %>>%  my_elasticnet_learner
)


set.seed(2024)
instance = tune(
  tuner = tnr("random_search", batch_size=20), ### tuning method
  #tuner = tnr("mbo"), ### tuning method
  task = task_train,
  learner = graph_learner_elastic_net,
  resampling = rsmp("cv", folds = 5), #### resampling method: 5-fold cross validation
  measures = msr("classif.ce"), #### classification error
  terminator = trm("evals", n_evals = 50) #### terminator
)

as.data.table(instance$archive$data)
instance$result_learner_param_vals


instance$result_y


######## result on test set

graph_learner_elastic_net$param_set$values = instance$result_learner_param_vals
graph_learner_elastic_net$param_set$values$classif.glmnet.lambda =graph_learner_elastic_net$param_set$values$classif.glmnet.s
graph_learner_elastic_net$train(task_train)
graph_learner_elastic_net$predict(task_test)$score()


### baseline with no features
baseline <- lrn("classif.featureless")$train(task_train)
baseline$predict(task_test)$score()


### see beta
graph_learner_elastic_net$model$classif.glmnet$model$beta



#################################
### c (Nested Cross Validation)



at = auto_tuner(
  tuner = tnr("random_search", batch_size=20),
  learner = graph_learner_elastic_net,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.ce"),
  terminator = trm("evals", n_evals = 50)
)

at$train(task_train)

set.seed(2024)
rr = resample(task, at , rsmp("cv", folds = 5))
rr
rr$score(msr("classif.ce"))[, .(iteration, classif.ce)]
rr$aggregate(msr("classif.ce"))

#################################
### Exercise BONUS (Nested Cross Validation Benchmark)

# Redefine the logistic regression Enet model
graph_learner_glmnet = (
  po("encode", method="treatment", affect_columns=selector_type("factor")) %>>%
    po("scale", affect_columns=selector_type("numeric")) %>>%
    lrn("classif.glmnet", s=to_tune(0,1), alpha=to_tune(0,1))
) %>% 
  as_learner() %>% 
  auto_tuner(
    tuner = tnr("random_search", batch_size=25),
    learner = .,
    resampling = rsmp("cv", folds=5),
    measure = msr("classif.ce"),
    terminator = trm("evals", n_evals=50)
  )

# Define a tree model on numerical features only
graph_learner_rpart_numeric = (
  po("select", affect_columns=selector_type("numeric")) %>>%
  lrn("classif.rpart", maxdepth = to_tune(1, 10))
  ) %>% 
  as_learner() %>% 
  auto_tuner(
    tuner = tnr("grid_search", batch_size=25),
    learner = .,
    resampling = rsmp("cv", folds=5),
    measure = msr("classif.ce")
  )

# Define a tree model on categorical features only
graph_learner_rpart_categorical = (
  po("select", affect_columns=selector_type("factor")) %>>%
    po("encodeimpact") %>>%
    lrn("classif.rpart", maxdepth = to_tune(1, 10))
) %>% 
  as_learner() %>% 
  auto_tuner(
    tuner = tnr("grid_search", batch_size=25),
    learner = .,
    resampling = rsmp("cv", folds=5),
    measure = msr("classif.ce")
  )

# Set up a benchmark of the different learners
bm_design = benchmark_grid(
  tasks = list(task),
  learners = list(logreg = graph_learner_glmnet,
                  tree_cat = graph_learner_rpart_categorical,
                  tree_num = graph_learner_rpart_numeric,
                  featureless = lrn("classif.featureless")
                  ),
  resamplings=list(rsmp("cv", folds=5))
)

bm = benchmark(
  design = bm_design
)

bm$aggregate()












