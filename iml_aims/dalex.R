library(mlr3)
library(mlr3learners )
library(mlr3tuning)
library(OpenML)
library(mlr3pipelines)
library(future)
library(DALEX)
library(DALEXtra)
#future::plan("multisession") 
library(ggplot2)
#future::plan("multicore")  ## doesn't work on Windows


# load bike  data 
bike_data = getOMLDataSet(data.id = 42713)

bike_data$data <- bike_data$data[,-c(7,13,14)] ## remove casual and registered as sum of the two is count. also remove working day due to collinearity.

### convert dates to factors 
bike_data$data$year <- factor(bike_data$data$year)
bike_data$data$month <- factor(bike_data$data$month)
bike_data$data$hour <- factor(bike_data$data$hour)
bike_data$data$weekday <- factor(bike_data$data$weekday)


bike_windspeed_data <- bike_data$data[,c("windspeed","count")]


# create  task
task = as_task_regr(bike_data$data, target = "count") 




my_ranger_learner = lrn("regr.ranger",
                     mtry.ratio = to_tune(0.1, 1),
                     min.node.size = to_tune(1, 50)
)
                     
                   

graph_learner_ranger = as_learner(
  po("encode", method = "treatment",
     affect_columns = selector_type("factor"), id = "binary_enc") %>>% po("scale") %>>% my_ranger_learner
)


at_ranger = auto_tuner(
  tuner = tnr("random_search", batch_size=1),
  learner = graph_learner_ranger,
  resampling = rsmp("cv", folds = 3),
  measure = msr("regr.mse"),
  terminator = trm("evals", n_evals = 1)
)

at_ranger$train(task)




ranger_exp = DALEXtra::explain_mlr3(at_ranger,
                                 data = bike_data$data[,-12],
                                 y = bike_data$data$data$count,
                                 label = "biking",
                                 colorize = FALSE)


######### SHAP

summer_evenning= bike_data$data[2375,-12]
summer_evenning

predict(ranger_exp, summer_evenning)
plot(predict_parts(ranger_exp, new_observation = summer_evenning))




plot(predict_parts(ranger_exp, new_observation = summer_evenning, B=25)) ### default N=all, B=10

######## ICE

random_rows_30 <- sample(1:nrow(bike_data$data), 30)

plot(predict_profile(ranger_exp, bike_data$data[random_rows_30,-12]))



############ pdp
ranger_profiles = model_profile(ranger_exp)

plot(ranger_profiles) +
  theme(legend.position = "top") +
  ggtitle("Partial Dependence for bike data","")



