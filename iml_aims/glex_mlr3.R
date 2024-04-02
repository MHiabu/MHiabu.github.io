library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(OpenML)
library(mlr3pipelines)
library(glex) #install.packages("glex", repos = "https://plantedml.r-universe.dev")



credit_data = getOMLDataSet(data.id = 31)
task = as_task_classif(credit_data$data, target = "class")



my_xgb_learner = lrn("classif.xgboost",
                     eta = to_tune(0, 0.5),
                     nrounds = to_tune(10, 5000),
                     max_depth = to_tune(1, 2),
                     predict_type = "prob")

preprocess1 <-  po("encode", method = "treatment",
                   affect_columns = selector_type("factor"), id = "binary_enc") %>>% 
                   po("scale")

preprocess2 <-   po("encode", method = "treatment",
                    affect_columns = selector_type("factor"), id = "binary_enc")

graph_learner_xg1 = as_learner(
  preprocess1 %>>% my_xgb_learner
)

graph_learner_xg2 = as_learner(
  preprocess2 %>>% my_xgb_learner
)


at_xg1 = auto_tuner(
  tuner = tnr("random_search", batch_size=1),
  learner = graph_learner_xg1,
  resampling = rsmp("cv", folds = 2),
  measure = msr("classif.ce"),
  terminator = trm("evals", n_evals = 2)
)


at_xg2 = auto_tuner(
  tuner = tnr("random_search", batch_size=1),
  learner = graph_learner_xg2,
  resampling = rsmp("cv", folds = 2),
  measure = msr("classif.ce"),
  terminator = trm("evals", n_evals = 2)
)


at_xg1$train(task)
at_xg2$train(task)





#### extract data from preprocessing



######## preprocess1
# preprocessing is done on the training data, but it could also be done on the test data

preprocess1$train(task)
my_data1 <- preprocess1$predict(task)$scale.output$data()

######## preprocess2

preprocess2$train(list(task))
my_data2 <- preprocess2$predict(list(task))$output$data()





#### extract xgboost model from autotuner
xgbooster1 <- at_xg1$model$learner$model$classif.xgboost$model
xgbooster2 <- at_xg2$model$learner$model$classif.xgboost$model



### now we can apply glex
my_expl1 <- glex(xgbooster1,as.matrix(my_data1[,-1]) )
my_expl2 <- glex(xgbooster2,as.matrix(my_data2[,-1]) )

### for how to use glex, see also here:
# https://github.com/PlantedML/glex and here:
# https://plantedml.com/glex/articles/Bikesharing-Decomposition-rpf.html
# note that glex may take quite long to calculate if 
# too many high order interactions are present.