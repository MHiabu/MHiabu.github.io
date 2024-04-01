library(gbm)
library(xgboost)
library(mlr3extralearners)


n <- 5000
set.seed(42)
x1 <- sample(0:1, n, replace=TRUE)
x2 <- sample(18:65, n, replace=TRUE)
x3 <- sample(0:1, n, replace=TRUE)
x4 <- sample(0:1, n, replace=TRUE)

x <- cbind(x1,x2,x3,x4)


getlambda <- function(x1,x2,x3,x4){
  0.2*(1+0.1*(x1==1))*(1+(1/(sqrt(x2-17))))*(1+(0.3*(18<=x2)*(x2<=35))*(x4==1)  -(0.3*(45<=x2)*(x2<=65))*(x4==1)  )
}

y <- sapply(1:n, function(i) rpois(1,getlambda(x1[i],x2[i],x3[i],x4[i])))



my_data = data.frame(x1=x1,x2=x2,x3=x3,x4=x4,y=y)
task = as_task_regr(my_data, target = "y") 


at = auto_tuner(
  tuner = tnr("random_search", batch_size=10),
  learner = lrn("regr.xgboost",
                eta = to_tune(0, 0.5),
                nrounds = to_tune(10, 5000),
                max_depth = to_tune(1, 3)),
  resampling = rsmp("cv", folds = 5),
  measure = msr("regr.mse"),
  terminator = trm("evals", n_evals = 50)
)

at$train(task)


xgb_exp = DALEXtra::explain_mlr3(at,
                                    data = my_data[,-5],
                                    y = my_data[,5],
                                    label = "risk",
                                    colorize = FALSE)




xgb_profiles = model_profile(xgb_exp)

plot(xgb_profiles) +
  theme(legend.position = "top") +
  ggtitle("Partial Dependence for risk data","")




at2 = auto_tuner(
  tuner = tnr("random_search", batch_size=5),
  learner = lrn("regr.gbm",
                shrinkage = to_tune(0, 0.5),
                n.trees = to_tune(10, 5000),
                interaction.depth = to_tune(1, 3),
                distribution="poisson"),
  resampling = rsmp("cv", folds = 5),
  measure = msr("regr.mse"),
  terminator = trm("evals", n_evals = 50)
)



at2$train(task)


gbm_exp = DALEXtra::explain_mlr3(at2,
                                 data = my_data[,-5],
                                 y = my_data[,5],
                                 label = "risk",
                                 colorize = FALSE)




gbm_profiles = model_profile(gbm_exp)

plot(gbm_profiles) +
  theme(legend.position = "top") +
  ggtitle("Partial Dependence for risk data","")


