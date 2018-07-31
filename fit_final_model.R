require(pheatmap)
require(ggbiplot)
require(ggplot2)
require(ggsci)
require(ggpubr)
require(lmerTest)
require(gbm)
require(reshape2)
require(stringr)
require(gridExtra)

# load function to clean the training data
source("clean_data.R")

# set up cross validation for tuning ML models
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2,
                           returnResamp = "final",
                           allowParallel = F)

# fitControl <- trainControl(method = "boot632",
#                            number = 25,
#                            #repeats = 2,
#                            returnResamp = "final",
#                            allowParallel = F)

# set of ML methods to compare
methods <- c("lm.vif",
             "pls",
             #'bayesglm',
             "glmnet",
             "ridge",
             #"lasso",
             'BstLm',
             "pcr",
             "icr",
             "gbm"
             #"nnet"
)

#__________________________________________________________
# full model to be applied to new data

full <- clean_data(raw = read.csv("full.csv"))
full.vif <- clean_data(raw = read.csv("full.csv"), correlation_threshold = 0.99, vif_threshold = 10)

# examine the degenerate features and the OLS linear model
plot(full.vif$subgraph, main = "approximately degenerate features")
plot(full.vif$ols, which = c(1:2,4:5), cook.levels = c(1,4), ask = F)

# fit ML models
finalmodel.cv <- sapply(methods, function(this_method){
    print(this_method)
    real_method <- this_method
    if(this_method == "lm.vif"){
        # limit collinearity using vif for lm
        X <- full.vif$X
        Y <- full.vif$Y$Dependent_Feature
        this_method <- "lm"
    } else {
        # don't limit collinearity for the rest
        X = full$X
        Y = full$Y$Dependent_Feature
    }

    # nnet needs to scale y between 0,1
    denom <- 1
    offset <- 0
    if(this_method == "nnet") {
        offset <- min(Y)
        denom <- max(Y-offset)
    }

    # train the model
    set.seed(137)
    model <- train(x = X,
                   y = (Y-offset)/denom,
                   method = this_method,
                   metric = 'RMSE',
                   tuneLength = 20,
                   trControl = fitControl)

    return(list(model = model,
                method = real_method,
                offset = offset,
                multiplier = denom))

}, USE.NAMES = TRUE, simplify = FALSE)


# print cv plots
for(ii in 2:length(finalmodel.cv)) print(plot(finalmodel.cv[[ii]]$model))


# compute training errors
training.errors.final <- sapply(finalmodel.cv, function(this_model){
    this_pred <- predict(this_model$model, full$X) * this_model$multiplier + this_model$offset
    data.frame(SUBID = rownames(full$X),
               Ytrue = full$Y[[1]],
               Ypred = this_pred,
               error = this_pred - full$Y[[1]],
               model = rep(this_model$method, length(this_pred)),
               dataset = rep("final", length(this_pred)))
}, simplify = F)
training.final.RMSE <- sapply(training.errors.final, function(x) sqrt(mean(x$error^2)), simplify = T)
training.final.RMSE.ordered <- sort(training.final.RMSE, decreasing = F)

# bind errors
cverrors.final <- do.call(rbind, training.errors.final)
cverrors.final$model <- factor(cverrors.final$model, levels = names(training.final.RMSE.ordered))



# look at cv fold results from training
rValues <- resamples(sapply(finalmodel.cv, function(this) this$model, USE.NAMES = T, simplify = F))
rValues.long <- melt(rValues$values, id.vars = "Resample")
splitnames <- str_split(rValues.long$variable, "~")
rValues.long$method <- sapply(splitnames, function(x) x[1])
rValues.long$metric <- sapply(splitnames, function(x) x[2])

# rescale the nnet
rValues.long$value[rValues.long$method == "nnet" & rValues.long$metric %in% c("MAE", "RMSE")] <- rValues.long$value[rValues.long$method == "nnet" & rValues.long$metric %in% c("MAE", "RMSE")] * finalmodel.cv$nnet$multiplier
rValues.long$method <- factor(rValues.long$method, levels = names(training.final.RMSE.ordered))

# order by the mean RMSE across folds
RMSE_sub <- subset(rValues.long, rValues.long$metric=="RMSE")
RMSE_subsum <- aggregate(RMSE_sub$value, list(RMSE_sub$method), FUN = mean)
RMSEsd_subsum <- aggregate(RMSE_sub$value, list(RMSE_sub$method), FUN = sd)
rValues.long$method <- factor(rValues.long$method, levels = factor(RMSE_subsum$Group.1[order(RMSE_subsum$x)]))

gg.final.cv <- ggplot(rValues.long) +
    aes(y = value, x = method) +
    geom_boxplot(aes(fill = method), alpha = 0.7, notch = T, notchwidth = .75) +
    geom_jitter(alpha = 0.4) +
    scale_color_d3() +
    scale_fill_d3(guide = F) +
    facet_grid(~metric, scales = "free_x") +
    coord_flip() +
    ggtitle("Error estimates from cross-validation folds") +
    theme_bw()

plot(gg.final.cv)
# summary(rValues)

RMSE.df <- data.frame(method = RMSE_subsum$Group.1,
           RMSE = RMSE_subsum$x,
           seRMSE = RMSEsd_subsum$x)

RMSE.df$method[order(RMSE.df$RMSE)]

#load("RMSE_summary.sorted.RData")
gg.final.error <- ggplot(cverrors.final[cverrors.final$model %in% RMSE.df$method[order(RMSE.df$RMSE)][1:5],]) +
    aes(x = Ytrue, y = Ypred, color = model, fill = model) +
    geom_point(size = 2) +
    geom_abline(slope = 1, intercept = 0, lty = 2) +
    stat_smooth(method = "lm") +
    ggtitle("Final model fit") +
    xlab("true value") +
    ylab("fit value") +
    scale_color_d3() +
    scale_fill_d3() +
    facet_grid(dataset~model) +
    guides(fill=FALSE, color=FALSE) +
    theme_bw()

gg.final.residual <- ggplot(cverrors.final[cverrors.final$model %in% RMSE.df$method[order(RMSE.df$RMSE)][1:5],]) +
    aes(x = Ytrue, y = Ypred - Ytrue, color = model, fill = model) +
    geom_point(size = 2) +
    geom_abline(slope = 0, intercept = 0, lty = 2) +
    stat_smooth(method = "lm") +
    ggtitle("Residuals") +
    xlab("true value") +
    ylab("error (fit-true)") +
    scale_color_d3() +
    scale_fill_d3() +
    facet_grid(dataset~model) +
    #facet_grid(model~dataset) +
    guides(fill=FALSE, color=FALSE) +
    theme_bw()

grid.arrange(gg.final.error, gg.final.residual, nrow = 2)




# compute variable importance
viplist <- lapply(finalmodel.cv, function(x) varImp(x$model, useModel = T)$importance)
vip <- as.data.frame(lapply(viplist, function(this) this[colnames(full$X),1,drop=F]),
                     row.names = colnames(full$X))
colnames(vip) = names(finalmodel.cv)
vip[is.na(vip)] <- 0

# plot variable importance
gg.vip <- pheatmap(vip[,as.character(RMSE.df$method[order(RMSE.df$RMSE)])],
         cluster_cols = F,
         main = "final model variable importance")

# gg.vip.transpose <- pheatmap(t(vip[,as.character(RMSE.df$method[order(RMSE.df$RMSE)])]),
#                    cluster_rows = T,
#                    main = "final model variable importance")

gg.vip.transpose <- pheatmap(t(vip[,c("pls", "lm.vif", "gbm", "glmnet", "pcr")]),
                             cluster_rows = T,
                             main = "final model variable importance")


save(finalmodel.cv,
     full,
     full.vif,
     vip,
     cverrors.final,
     gg.vip,
     gg.vip.transpose,
     gg.final.error,
     gg.final.residual,
     gg.final.cv,
     training.final.RMSE,
     rValues,
     rValues.long,
     RMSE.df,
     file = "final_models.RData")







