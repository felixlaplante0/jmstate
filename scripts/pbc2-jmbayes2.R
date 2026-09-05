suppressPackageStartupMessages({
    library(JMbayes2)
    library(nlme)
    library(survival)
})

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
root <- normalizePath(file.path(dirname(script_path), ".."))
results_dir <- file.path(root, "results")
dir.create(results_dir, showWarnings = FALSE)

seed <- 42L
n_iter <- 3500L
n_burnin <- 500L
n_prediction_samples <- 100L
n_prediction_mcmc <- 10L

pbc2 <- read.csv(file.path(root, "data", "pbc2.csv"))
pbc2 <- pbc2[order(pbc2$id, pbc2$year), ]
pbc2$year10 <- pbc2$year / 10
pbc2$years10 <- pbc2$years / 10
pbc2$log_bilirubin <- log(pbc2$serBilir)
pbc2$drug01 <- as.numeric(pbc2$drug == "D-penicil")

survival_data <- pbc2[!duplicated(pbc2$id), c("id", "years10", "status2", "drug01", "age")]
survival_data$age_z <- as.numeric(scale(survival_data$age))
pbc2$age_z <- survival_data$age_z[match(pbc2$id, survival_data$id)]

prothrombin_mean <- mean(pbc2$prothrombin, na.rm = TRUE)
prothrombin_sd <- sd(pbc2$prothrombin, na.rm = TRUE)
pbc2$prothrombin_z <- (pbc2$prothrombin - prothrombin_mean) / prothrombin_sd
prothrombin_paths <- split(pbc2[c("year10", "prothrombin_z")], pbc2$id)

prothrombin_at <- function(time, id, cutoff) {
    mapply(function(t, subject, available_until) {
        path <- prothrombin_paths[[as.character(subject)]]
        path <- path[path$year10 <= available_until & !is.na(path$prothrombin_z), , drop = FALSE]
        approx(path$year10, path$prothrombin_z, xout = t, rule = 2, ties = "ordered")$y
    }, time, id, cutoff)
}

pbc2$cutoff <- ave(pbc2$year10, pbc2$id, FUN = max)
subject_ids <- sort(unique(pbc2$id))
fold_table <- read.csv(file.path(results_dir, "pbc2-cv-folds.csv"))
fold_table <- fold_table[match(subject_ids, fold_table$id), ]
fold <- as.integer(fold_table$fold)
fold_indices <- sort(unique(fold))

grid <- read.csv(file.path(results_dir, "pbc2-prediction-grid.csv"))
landmarks <- sort(unique(grid$landmark))
horizons_by_landmark <- lapply(landmarks, function(landmark) {
    sort(grid$horizon[abs(grid$landmark - landmark) < 1e-10])
})

fit_model <- function(ids, baseline, chains, fit_seed) {
    longitudinal <- pbc2[pbc2$id %in% ids, ]
    event <- survival_data[survival_data$id %in% ids, ]
    mixed <- nlme::lme(
        log_bilirubin ~ year10 + prothrombin_at(year10, id, cutoff),
        random = list(id = nlme::pdDiag(~ year10)),
        data = longitudinal,
        control = nlme::lmeControl(opt = "optim")
    )
    event_model <- survival::coxph(
        Surv(years10, status2) ~ drug01 + age_z,
        data = event,
        x = TRUE,
        model = TRUE
    )
    control <- list(
        n_chains = chains,
        n_iter = n_iter,
        n_burnin = n_burnin,
        seed = fit_seed,
        cores = chains,
        parallel = "multicore"
    )
    priors <- list(penalty_alphas = "none", penalty_gammas = "none")
    if (baseline == "spline") {
        control <- c(
            control,
            list(
                basis = "bs",
                Bsplines_degree = 2L,
                base_hazard_segments = 10L,
                timescale_base_hazard = "identity"
            )
        )
        priors$penalized_bs_gammas <- FALSE
    }
    JMbayes2::jm(
        event_model,
        mixed,
        time_var = "year10",
        functional_forms = ~ value(log_bilirubin),
        base_hazard = if (baseline == "parametric") "weibull" else NULL,
        data_Surv = event,
        id_var = "id",
        control = control,
        priors = priors
    )
}

parameter_summary <- function(fit, baseline) {
    means <- fit$statistics$Mean
    blocks <- intersect(c("betas", "betas_HC", "sigmas", "D", "bs_gammas", "gammas", "alphas"), names(means))
    do.call(rbind, lapply(blocks, function(block) {
        estimate <- means[[block]]
        data.frame(
            baseline = baseline,
            block = block,
            parameter = names(estimate),
            estimate = as.numeric(estimate),
            posterior_sd = as.numeric(fit$statistics$SD[[block]]),
            lower_95 = as.numeric(fit$statistics$CI_low[[block]]),
            upper_95 = as.numeric(fit$statistics$CI_upp[[block]]),
            effective_size = as.numeric(fit$statistics$Effective_Size[[block]]),
            row.names = NULL
        )
    }))
}

predict_survival <- function(fit, ids, landmark, horizons, prediction_seed) {
    event <- survival_data[survival_data$id %in% ids, ]
    at_risk_ids <- event$id[event$years10 > landmark]
    newdata <- pbc2[pbc2$id %in% at_risk_ids & pbc2$year10 <= landmark, ]
    newdata$cutoff <- pmin(newdata$cutoff, landmark)
    newdata$years10 <- landmark
    newdata$status2 <- 0
    future <- horizons[horizons > landmark]
    survival <- matrix(1, nrow = length(at_risk_ids), ncol = length(horizons), dimnames = list(at_risk_ids, horizons))
    prediction <- predict(
        fit,
        newdata = newdata,
        times = future,
        process = "event",
        return_newdata = TRUE,
        n_samples = n_prediction_samples,
        n_mcmc = n_prediction_mcmc,
        cores = 1L,
        seed = prediction_seed
    )
    prediction <- prediction[prediction$years10 > landmark, c("id", "years10", "pred_CIF")]
    row_index <- match(prediction$id, at_risk_ids)
    column_index <- match(prediction$years10, horizons)
    survival[cbind(row_index, column_index)] <- 1 - prediction$pred_CIF
    rows <- expand.grid(id = at_risk_ids, horizon = horizons, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
    rows$survival <- as.vector(t(survival))
    rows
}

full_fits <- list()
fit_stats <- list()
for (baseline in c("parametric", "spline")) {
    cat(sprintf("Fitting full-data JMbayes2 %s baseline...\n", baseline))
    started <- proc.time()[["elapsed"]]
    fit <- fit_model(subject_ids, baseline, chains = 3L, fit_seed = seed)
    elapsed <- proc.time()[["elapsed"]] - started
    full_fits[[baseline]] <- fit
    write.csv(parameter_summary(fit, baseline), file.path(results_dir, sprintf("pbc2-jmbayes2-%s-parameters.csv", baseline)), row.names = FALSE)
    values <- unlist(fit$fit_stats)
    values <- values[grepl("\\.(DIC|pD|LPML|WAIC)$", names(values))]
    fit_stats[[baseline]] <- data.frame(baseline = baseline, statistic = names(values), value = as.numeric(values))
    fit_stats[[baseline]] <- rbind(fit_stats[[baseline]], data.frame(baseline = baseline, statistic = "elapsed_seconds", value = elapsed))
}
write.csv(do.call(rbind, fit_stats), file.path(results_dir, "pbc2-jmbayes2-fit-stats.csv"), row.names = FALSE)

prediction_rows <- list()
timing_rows <- list()
for (baseline in c("parametric", "spline")) {
    for (fold_index in fold_indices) {
        cat(sprintf("Fitting JMbayes2 %s fold %d...\n", baseline, fold_index))
        train_ids <- subject_ids[fold != fold_index]
        test_ids <- subject_ids[fold == fold_index]
        fit_started <- proc.time()[["elapsed"]]
        fit <- fit_model(train_ids, baseline, chains = 1L, fit_seed = seed + fold_index)
        fit_elapsed <- proc.time()[["elapsed"]] - fit_started
        prediction_started <- proc.time()[["elapsed"]]
        for (landmark_index in seq_along(landmarks)) {
            landmark <- landmarks[landmark_index]
            rows <- predict_survival(fit, test_ids, landmark, horizons_by_landmark[[landmark_index]], seed + 100L * fold_index + landmark_index)
            rows$baseline <- baseline
            rows$fold <- fold_index
            rows$landmark <- landmark
            prediction_rows[[length(prediction_rows) + 1L]] <- rows
        }
        prediction_elapsed <- proc.time()[["elapsed"]] - prediction_started
        timing_rows[[length(timing_rows) + 1L]] <- data.frame(baseline = baseline, fold = fold_index, fit_seconds = fit_elapsed, prediction_seconds = prediction_elapsed)
    }
}
write.csv(do.call(rbind, prediction_rows), file.path(results_dir, "pbc2-jmbayes2-predictions.csv"), row.names = FALSE)
write.csv(do.call(rbind, timing_rows), file.path(results_dir, "pbc2-jmbayes2-timings.csv"), row.names = FALSE)
cat("JMbayes2 PBC2 comparison outputs written.\n")
