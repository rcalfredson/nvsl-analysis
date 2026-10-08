# Independent REML reference for test/test_learner_metric_stats.py::observations.
# Input: the model_input CSV exported by analyze_repeated_metrics for that fixture.
# Usage: Rscript test/reference/learner_metric_mixed_nlme.R INPUT.csv OUTPUT_DIR
# nlme supplies a separate random-intercept implementation. Its default t tests
# are intentionally not used: the Python report specifies asymptotic Wald tests.
library(nlme)
args <- commandArgs(trailingOnly=TRUE)
data <- read.csv(args[1])
data$cohort <- factor(data$cohort, levels=c("weak", "strong"))
data$radius_pair <- factor(data$radius_pair, levels=0:2)
for (metric in unique(data$metric)) {
  fit <- lme(value ~ cohort * radius_pair, random=~1|unit_id,
             data=data[data$metric == metric,], method="REML")
  write.table(t(fixef(fit)), file=file.path(args[2], paste0(metric, "_r_beta.csv")),
              sep=",", row.names=FALSE, col.names=FALSE)
  write.table(vcov(fit), file=file.path(args[2], paste0(metric, "_r_covariance.csv")),
              sep=",", row.names=FALSE, col.names=FALSE)
}
cat(R.version.string, "\nnlme", as.character(packageVersion("nlme")), "\n")
