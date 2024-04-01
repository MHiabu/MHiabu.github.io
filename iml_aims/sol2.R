
library(tidyverse)
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(OpenML)
library(mlr3pipelines)

bike_data = getOMLDataSet(data.id = 42713)
bike_data$data <- bike_data$data[,-c(7,13,14)] ## remove casual and registered as sum of the two is count. also remove working day due to collinearity.

### convert dates to factors 
bike_data$data$year <- factor(bike_data$data$year)
bike_data$data$month <- factor(bike_data$data$month)
bike_data$data$hour <- factor(bike_data$data$hour)
bike_data$data$weekday <- factor(bike_data$data$weekday)

# a
task_count_on_windspeed = as_task_regr(
  x = bike_data$data %>% select("windspeed", "count"),
  target="count"
)

lm_count_on_windspeed = lrn("regr.lm")
lm_count_on_windspeed$train(task_count_on_windspeed)
lm_count_on_windspeed$model$coefficients
# it seems high windspeeds has a positive effect on bike rental (beta= 2.063179).
# But this may be due to confounders and we should rather fit the whole data

# b

fac_enc = po(
  "encode",
  method="treatment",
  affect_columns=selector_type("factor")
  )

task_count_on_all = as_task_regr(
  x = bike_data$data,
  target="count"
)

lm_count_on_all = (
  fac_enc %>>%
    lrn("regr.lm")
) %>% as_learner()

lm_count_on_all$train(
  task_count_on_all
)

b_coef = lm_count_on_all$model$regr.lm$model$coefficients[["windspeed"]]
# now the effect of windspeed is more realistic.
# windspeed has an estimated beta of -0.4353277 

# c

# count ~ (X - windspeed)
task_count_on_remaining = as_task_regr(
  x = bike_data$data %>% select(-c("windspeed")),
  target = "count"
)
lm_count_on_remaining = (
  fac_enc %>>%
  lrn("regr.lm")
  ) %>% 
  as_learner()
lm_count_on_remaining$train(task_count_on_remaining)
res_on_remaining = lm_count_on_remaining$model$regr.lm$model$residuals

# windspeed ~ (X-windspeed)
task_windspeed_on_remaining = as_task_regr(
  x=bike_data$data %>% select(-c("count")),
  target="windspeed"
)
lm_windspeed_on_remaining = (
  fac_enc %>>%
    lrn("regr.lm")
  ) %>% 
  as_learner()
lm_windspeed_on_remaining$train(task_windspeed_on_remaining)
windspeed_res_on_remaining = lm_windspeed_on_remaining$model$regr.lm$model$residuals

# count_residual ~ windspeed_residual
task_count_res_on_windspeed_res = data.frame(
  windspeed = windspeed_res_on_remaining,
  count = res_on_remaining
) %>% 
  as_task_regr(
    target="count"
  )
lm_count_res_on_windspeed_res = (
  fac_enc %>>%
    lrn("regr.lm")
) %>% 
  as_learner()
lm_count_res_on_windspeed_res$train(task_count_res_on_windspeed_res)
c_coef = lm_count_res_on_windspeed_res$model$regr.lm$model$coefficients[["windspeed"]]

# d
b_coef - c_coef # small on the order of rounding errors
# Hence we get the same from the two procedures

# e
autoknn_windspeed_on_remaining = (
  fac_enc %>>%
    auto_tuner(
      tuner=tnr("grid_search", resolution=8),
      learner=lrn("regr.kknn", k=to_tune(10, 5000)),
      resampling=rsmp("cv", folds=3),
      terminator=trm("evals", n_evals=20)
    )
) %>% 
  as_learner()

autoknn_windspeed_on_remaining$train(task_count_res_on_windspeed_res)

autoknn_windspeed_on_remaining$model$regr.kknn.tuned$model

# Compare results
lm_pred = lm_count_res_on_windspeed_res$predict(task_count_res_on_windspeed_res)
knn_pred = autoknn_windspeed_on_remaining$predict(task_count_res_on_windspeed_res)

res = data.frame(
  windspeed_res = task_count_res_on_windspeed_res$data()$windspeed,
  lm = lm_pred$response,
  knn = knn_pred$response
) %>% 
  pivot_longer(cols=-c(windspeed_res)) 
(res %>% 
  ggplot(aes(x = windspeed_res, y = value, color=name)) +
  geom_line() +
  geom_point(alpha=0)
) %>% 
  ggExtra::ggMarginal(margins = "x")

resdata = task_count_res_on_windspeed_res$data() %>% as.data.frame()
resdata %>% 
  ggplot(aes(x =windspeed, y = count)) +
  geom_point() +
  geom_smooth(method = "gam", color = "red", se=F) +
  geom_smooth(method="lm", se=F) +
  coord_cartesian(ylim=c(-20,20))

# Benchmarking linear model against knn -------------------------------
# Ideally we should include the autotuned knn in the benchmark,
# but that is a lot slower.

lrn_1 = (
  fac_enc %>>%
    lrn("regr.kknn", k = 5)
) %>% 
  as_learner()
lrn_1$id = "knn_small_k"

lrn_2 = (
  fac_enc %>>%
    lrn("regr.kknn", k = 1435)
) %>% 
  as_learner()
lrn_2$id = "knn_big_k"

lrn_3 = (
  fac_enc %>>%
    lrn("regr.lm")
) %>% 
  as_learner()
lrn_3$id = "lm"

lrn_4 = (
  fac_enc %>>%
    lrn("regr.featureless")
) %>% 
  as_learner()
lrn_4$id = "featureless"

design = benchmark_grid(
  tasks=task_count_res_on_windspeed_res,
  learners = list(lrn_1, lrn_2, lrn_3, lrn_4),
  resamplings = rsmp("cv", folds=5)
  )

bm = benchmark(design)
bm$aggregate()
# a well-tuned knn seems to outperform lm slightly
# lm is better than featureless (thankfully)
# a poorly tuned knn is very, very bad

