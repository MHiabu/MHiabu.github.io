library(mlr3)
library(mlr3learners )
library(mlr3tuning)
library(OpenML)
library(mlr3pipelines)
library(future)

future::plan("multisession") 
#future::plan("multicore")  ## doesn't work on Windows



#################################
####### Exercise I 

# load credit-g data and define task
credit_data = getOMLDataSet(data.id = 31)
task = as_task_classif(credit_data$data, target = "class") 


#### binary encoding
ranger_binary = as_learner(
  po("encode", method = "treatment",
     affect_columns = selector_type("factor"), id = "binary_enc") %>>% lrn("classif.ranger",
                                                                           mtry.ratio = to_tune(0.1, 1),
                                                                           min.node.size = to_tune(1, 50),
                                                                           predict_type = "prob")
)
at_ranger_binary = auto_tuner(
  tuner = tnr("random_search", batch_size=20),
  learner = ranger_binary,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc"),
  terminator = trm("evals", n_evals = 50)
)


#### target encoding
ranger_tagret_po = as_learner(
  po("encodeimpact", affect_columns = selector_type("factor"), id = "target_enc") %>>% lrn("classif.ranger",
                                                                                           mtry.ratio = to_tune(0.1, 1),
                                                                                           min.node.size = to_tune(1, 50),
                                                                                           predict_type = "prob")
)
at_ranger_tagret_po = auto_tuner(
  tuner = tnr("random_search", batch_size=20),
  learner = ranger_tagret_po,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc"),
  terminator = trm("evals", n_evals = 50)
)

#### target encoding within ranger
ranger_target_intern = as_learner(
  lrn("classif.ranger",
      respect.unordered.factors = "order",
      mtry.ratio = to_tune(0.1, 1),
      min.node.size = to_tune(1, 50),
      predict_type = "prob")
)
at_ranger_target_intern = auto_tuner(
  tuner = tnr("random_search", batch_size=20),
  learner = ranger_target_intern,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc"),
  terminator = trm("evals", n_evals = 50)
)

#### target encoding within ranger for every split
ranger_target_persplit = as_learner(
  lrn("classif.ranger", 
      respect.unordered.factors = "partition",
      mtry.ratio = to_tune(0.1, 1),
      min.node.size = to_tune(1, 50),
      predict_type = "prob")
)
at_ranger_target_persplit = auto_tuner(
  tuner = tnr("random_search", batch_size=20),
  learner = ranger_target_persplit,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc"),
  terminator = trm("evals", n_evals = 50)
)


#### glmm encoding
ranger_glmm = as_learner(
  po("encodelmer", affect_columns = selector_type("factor"), id = "glmm_enc") %>>% lrn("classif.ranger",
                                                                                           mtry.ratio = to_tune(0.1, 1),
                                                                                           min.node.size = to_tune(1, 50),
                                                                                           predict_type = "prob")
)
at_ranger_glmm = auto_tuner(
  tuner = tnr("random_search", batch_size=20),
  learner = ranger_glmm,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc"),
  terminator = trm("evals", n_evals = 50)
)


########### verify ranger internal external same

baseline <- lrn("classif.featureless", predict_type = "prob")

start.time <- Sys.time()
res1 <- benchmark(
  benchmark_grid(
    task        = list(task),
    learners    = list(baseline),
    resamplings = list(rsmp("cv", folds = 3))
  ), store_models = TRUE)
end.time <- Sys.time()
time.taken1 <- end.time - start.time
time.taken1


start.time <- Sys.time()
res2 <- benchmark(
  benchmark_grid(
    task        = list(task),
    learners    = list( at_ranger_binary),
    resamplings = list(rsmp("cv", folds = 3))
  ), store_models = TRUE)
end.time <- Sys.time()
time.taken2 <- end.time - start.time
time.taken2


start.time <- Sys.time()
res3 <- benchmark(
  benchmark_grid(
    task        = list(task),
    learners    = list(at_ranger_glmm),
    resamplings = list(rsmp("cv", folds = 3))
  ), store_models = TRUE)
end.time <- Sys.time()
time.taken3 <- end.time - start.time
time.taken3


start.time <- Sys.time()
res4 <- benchmark(
  benchmark_grid(
    task        = list(task),
    learners    = list(at_ranger_tagret_po),
    resamplings = list(rsmp("cv", folds = 3))
  ), store_models = TRUE)
end.time <- Sys.time()
time.taken4 <- end.time - start.time
time.taken4



start.time <- Sys.time()
res55 <- benchmark(
  benchmark_grid(
    task        = list(task),
    learners    = list(at_ranger_target_intern),
    resamplings = list(rsmp("cv", folds = 3))
  ), store_models = TRUE)
end.time <- Sys.time()
time.taken55 <- end.time - start.time
time.taken55


start.time <- Sys.time()
res6 <- benchmark(
  benchmark_grid(
    task        = list(task),
    learners    = list( at_ranger_target_persplit),
    resamplings = list(rsmp("cv", folds = 3))
  ), store_models = TRUE)
end.time <- Sys.time()
time.taken6 <- end.time - start.time
time.taken6

# 
# res <- benchmark(
#   benchmark_grid(
#     task        = list(task),
#     learners    = list(baseline,
#                        at_ranger_binary,
#                        at_ranger_glmm,
#                        at_ranger_tagret_po,
#                        at_ranger_target_intern,
#                        at_ranger_target_persplit),
#     resamplings = list(rsmp("cv", folds = 3))
#   ), store_models = TRUE)




print(cbind(res1$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr"))), time=time.taken1))



print(cbind(res2$aggregate(list(msr("classif.ce"),
                                msr("classif.acc"),
                                msr("classif.auc"),
                                msr("classif.fpr"),
                                msr("classif.fnr"))), time=time.taken2))


print(cbind(res3$aggregate(list(msr("classif.ce"),
                                msr("classif.acc"),
                                msr("classif.auc"),
                                msr("classif.fpr"),
                                msr("classif.fnr"))), time=time.taken3))


print(cbind(res4$aggregate(list(msr("classif.ce"),
                                msr("classif.acc"),
                                msr("classif.auc"),
                                msr("classif.fpr"),
                                msr("classif.fnr"))), time=time.taken4))


print(cbind(res5$aggregate(list(msr("classif.ce"),
                                msr("classif.acc"),
                                msr("classif.auc"),
                                msr("classif.fpr"),
                                msr("classif.fnr"))), time=time.taken5))



print(cbind(res6$aggregate(list(msr("classif.ce"),
                                msr("classif.acc"),
                                msr("classif.auc"),
                                msr("classif.fpr"),
                                msr("classif.fnr"))), time=time.taken6))


### Ranger with internal target encoding seems to do quite well here, especially given the speed


#################################
######################### Exercise II

data2 <- credit_data$data ### create a copy that I can modify

####### modify data2. this does not need to be part of the graph as I am not using information on feature values

str(data2)
levels(data2$checking_status)

aggregate(class ~ checking_status, data2, function(x) mean(as.numeric(x)-1)) ### check order in target encoding

data2$checking_status <- as.numeric(factor(data2$checking_status, levels=c("no checking","<0", "0<=X<200",">=200"))) ### manual encoding


levels(data2$credit_history)

aggregate(class ~ credit_history, data2, function(x) mean(as.numeric(x)-1)) ### check order in target encoding, decision: leave as is

levels(data2$purpose)
aggregate(class ~ purpose, data2, function(x) mean(as.numeric(x)-1)) ### check order in target encoding, decision: leave as is



levels(data2$savings_status)
aggregate(class ~ savings_status, data2, function(x) mean(as.numeric(x)-1)) ### check order in target encoding, decision: leave as is
data2$savings_status <- as.numeric(factor(data2$savings_status, levels=c("no known savings",
                                                                          "<100",
                                                                         "100<=X<500",
                                                                         "500<=X<1000",
                                                                         ">=1000"
                                                                         ))) ### manual encoding


levels(data2$employment)
aggregate(class ~ employment, data2, function(x) mean(as.numeric(x)-1)) ### check order in target encoding, decision: leave as is
data2$employment <- as.numeric(factor(data2$employment, levels=c("unemployed",
                                                                          "<1",
                                                                          "1<=X<4",
                                                                          "4<=X<7",
                                                                          ">=7"))) ### manual encoding



##################### define graphlearner


ranger_myencoding = as_learner(
  po("removeconstants") %>>%
    po("collapsefactors", no_collapse_above_prevalence = 0.01) %>>% lrn("classif.ranger",
                                                                                         respect.unordered.factors = "order",
                                                                                         mtry.ratio = to_tune(0.1, 1),
                                                                                         min.node.size = to_tune(1, 50),
                                                                                         predict_type = "prob")
)
  
  
at_ranger_myencoding = auto_tuner(
  tuner = tnr("random_search", batch_size=20),
  learner = ranger_myencoding,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc"),
  terminator = trm("evals", n_evals = 50)
)




task = as_task_classif(data2, target = "class") 


start.time <- Sys.time()
res <- benchmark(
  benchmark_grid(
    task        = list(task),
    learners    = list( at_ranger_myencoding),
    resamplings = list(rsmp("cv", folds = 3))
  ), store_models = TRUE)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

print(cbind(res$aggregate(list(msr("classif.ce"),
                                msr("classif.acc"),
                                msr("classif.auc"),
                                msr("classif.fpr"),
                                msr("classif.fnr"))), time=time.taken))

##### seems to give no improvement compared to previous strategies. Ranger with internal target encoding seems to do quite well

