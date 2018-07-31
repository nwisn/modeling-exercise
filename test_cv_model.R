require(pheatmap)
require(ggbiplot)
require(ggplot2)
require(ggsci)
require(ggpubr)
require(lmerTest)
require(gbm)
require(reshape2)
require(stringr)
require(e1071)
require(gridExtra)

# load function to clean the training data
source("clean_data.R")

# set up cross validation for tuning ML models
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2,
                           returnResamp = "final",
                           allowParallel = F)

# fitControl <- trainControl(method = "optimism_boot",
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


#___________________________________________________________________
# fit and validate models

# get split training and testing sets
training <- clean_data(raw = read.csv("training.csv"), correlation_threshold = 1)
training.vif <- clean_data(raw = read.csv("training.csv"), correlation_threshold = .99, vif_threshold = 10)
testing.raw <- read.csv("testing.csv")

# examine the degenerate features and the OLS linear model
plot(training.vif$subgraph, main = "approximately degenerate features")
plot(training.vif$ols, which = c(1:2,4:5), cook.levels = c(1,4), ask = F)

# test assumptions about heteroskedasticity and autocorrelation of residuals
training.vif$skedastic
training.vif$autocor


# train and tune the ML models
model.cv <- sapply(methods, function(this_method){
    cat(paste("fitting", this_method, "\n"))

    # deal with the lm.vif exception
    real_method <- this_method
    if(this_method == "lm.vif"){
        # limit collinearity using vif for lm
        X <- training.vif$X
        Y <- training.vif$Y$Dependent_Feature
        this_method <- "lm"
    } else {
        # don't limit collinearity for the rest
        X = training$X
        Y = training$Y$Dependent_Feature
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
                   y = (Y - offset)/denom,
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
for(ii in 2:length(model.cv)) print(plot(model.cv[[ii]]$model))

# compute training errors
training.errors <- sapply(model.cv, function(this_model){
    this_pred <- predict(this_model$model, training$X) * this_model$multiplier + this_model$offset
    data.frame(SUBID = factor(rownames(training$X)),
               Ytrue = training$Y[[1]],
               Ypred = this_pred,
               error = this_pred - training$Y[[1]],
               model = rep(this_model$method, length(this_pred)),
               dataset = rep("training", length(this_pred)))
}, simplify = F)
training.RMSE <- sapply(training.errors, function(x) sqrt(mean(x$error^2)), simplify = F)
training.RMSEMAE <- as.data.frame(t(sapply(training.errors, function(x) postResample(x$Ypred, x$Ytrue), simplify = T)))

# compute testing errors
testing <- predict(training$preProcess.model, testing.raw) # apply preProcessing model
testing.errors <- sapply(model.cv, function(this_model){
    this_pred <- predict(this_model$model, testing)* this_model$multiplier + this_model$offset
    data.frame(SUBID = factor(testing$SUBID),
               Ytrue = testing$Dependent_Feature,
               Ypred = this_pred,
               error = this_pred - testing$Dependent_Feature,
               model = rep(this_model$method, length(this_pred)),
               dataset = rep("testing", length(this_pred)))
}, simplify = F)
testing.RMSE <- sapply(testing.errors, function(x) sqrt(mean(x$error^2)), simplify = F)
testing.RMSEMAE <- as.data.frame(t(sapply(testing.errors, function(x) postResample(x$Ypred, x$Ytrue), simplify = T)))

RMSE_summary <- as.data.frame(t(rbind(as.data.frame(training.RMSE, row.names = "training RMSE"),
                                      as.data.frame(testing.RMSE, row.names = "testing RMSE"))))
RMSE_summary.sorted <- RMSE_summary[order(RMSE_summary$`testing RMSE`),]
save(RMSE_summary.sorted, file = "RMSE_summary.sorted.RData")
print(RMSE_summary.sorted)

RMSEMAE_summary <- cbind(training.RMSEMAE, testing.RMSEMAE)
colnames(RMSEMAE_summary)[1:3] <- paste("training", colnames(RMSEMAE_summary)[1:3])
colnames(RMSEMAE_summary)[4:6] <- paste("testing", colnames(RMSEMAE_summary)[4:6])
RMSEMAE_summary.sorted <- RMSEMAE_summary[order(RMSEMAE_summary$`testing RMSE`),]

# bind errors
errors <- rbind(do.call(rbind, training.errors), do.call(rbind, testing.errors))
#errors$model <- factor(errors$model, levels = names(sort(unlist(testing.RMSE))))




#____________
# plot cv error estimates

# look at cv fold results from training
rValues <- resamples(sapply(model.cv, function(this) this$model, USE.NAMES = T, simplify = F))
rValues.long <- melt(rValues$values, id.vars = "Resample")
splitnames <- str_split(rValues.long$variable, "~")
rValues.long$method <- sapply(splitnames, function(x) x[1])
rValues.long$metric <- sapply(splitnames, function(x) x[2])

# rescale the nnet
rValues.long$value[rValues.long$method == "nnet" & rValues.long$metric %in% c("MAE", "RMSE")] <- rValues.long$value[rValues.long$method == "nnet" & rValues.long$metric %in% c("MAE", "RMSE")] * model.cv$nnet$multiplier

# sort by RMSE
RMSE_sub <- subset(rValues.long, rValues.long$metric=="RMSE")
RMSE_subsum <- aggregate(RMSE_sub$value, list(RMSE_sub$method), FUN = mean)
RMSEsd_subsum <- aggregate(RMSE_sub$value, list(RMSE_sub$method), FUN = sd)

RMSE.df <- data.frame(method = RMSE_subsum$Group.1,
                      RMSE = RMSE_subsum$x,
                      seRMSE = RMSEsd_subsum$x)

#rValues.long$method <- factor(rValues.long$method, levels = RMSE_subsum$Group.1[order(RMSE_subsum$x)])
rValues.long$method <- factor(rValues.long$method, levels = RMSE.df$method[order(RMSE.df$RMSE)])
errors$model <- factor(errors$model, levels = RMSE.df$method[order(RMSE.df$RMSE)])

gg.cv <- ggplot(rValues.long) +
    aes(y = value, x = method) +
    geom_boxplot(aes(fill = method), alpha = 0.7, notch = T, notchwidth = .75) +
    geom_jitter(alpha = 0.4) +
    scale_color_d3() +
    scale_fill_d3(guide = F) +
    facet_grid(~metric, scales = "free_x") +
    coord_flip() +
    ggtitle("Error estimates from cross-validation folds") +
    theme_bw()

print(gg.cv)
# summary(rValues)

# agg <- aggregate(rValues.long$value, list(paste(rValues.long$method, rValues.long$metric, sep=";")), FUN = mean)
# agg$method <- sapply(str_split(agg$Group.1, ";"), function(x) x[1])
# agg$metric <- sapply(str_split(agg$Group.1, ";"), function(x) x[2])
#
# unlist(testing.RMSE)
# aggRMSE <- agg[agg$metric=="RMSE",]
# aggRMSE[match(methods, aggRMSE$method),]
#
# plot(unlist(testing.RMSE), aggRMSE[match(methods, aggRMSE$method),"x"])
# abline(0,1)
#____________
# plot real errors


# template for pairwise comparisons
my_comparisons <- lapply(levels(errors$model)[-1], function(x) c(x, levels(errors$model)[1]))

# compare mean abs(error) across models with a mixed-effect model
testing.mem <- lmer(abs(error) ~ model + (1|SUBID), data = subset(errors, errors$dataset == "testing"))
mem.anova <- anova(testing.mem)
mem.summary <- summary(testing.mem)
fwer <- p.adjust(mem.summary$coefficients[-1,5], method = "bonferroni")
names(fwer) <- levels(errors$model)[-1]

# plot abs(error)
boxp <- ggplot(subset(errors, errors$dataset == "testing")) +
    aes(y = abs(error), x = model) +
    geom_boxplot(aes(fill = model), alpha = 0.7, notch = T, notchwidth = .75) +
    geom_point(alpha = 0.4) +
    geom_line(aes(group = SUBID), alpha = 0.1) +
    stat_compare_means(method = "anova", label.y = 45, label.x = 6) +
    stat_compare_means(comparisons = my_comparisons,
                       method = "t.test",
                       paired = T)+ # Add pairwise comparisons p-value
    scale_color_discrete(guide = F) +
    scale_linetype_discrete(guide = F) +
    scale_fill_d3(guide = F) +
    ggtitle("Absolute errors of test predictions") +
    theme_classic2()
## build plot
boxp2 <- ggplot_build(boxp)
## replace annotation with adjusted p-value
boxp2$data[[4]]$label <- paste0("Mixed-effect p = ", signif(mem.anova$`Pr(>F)`, 3))
boxp2$data[[5]]$annotation <- signif(fwer[sapply(str_split(boxp2$data[[5]]$group, "-"), function(x) x[1])],2)
absolute_error.plot <- boxp2
plot(ggplot_gtable(absolute_error.plot))
# end plot errors____________


# RMSE_summary <- as.data.frame(t(rbind(as.data.frame(training.RMSE, row.names = "training RMSE"),
#                                       as.data.frame(testing.RMSE, row.names = "testing RMSE"))))
# RMSE_summary.sorted <- RMSE_summary[order(RMSE_summary$`testing RMSE`),]
# save(RMSE_summary.sorted, file = "RMSE_summary.sorted.RData")
# print(RMSE_summary.sorted)

errors$model <- factor(errors$model, levels = RMSE.df$method[order(RMSE.df$RMSE)])
gg.error <- ggplot(errors[errors$model %in% RMSE.df$method[order(RMSE.df$RMSE)],]) +
    aes(x = Ytrue, y = Ypred, color = model, fill = model) +
    geom_point(size = 2) +
    geom_abline(slope = 1, intercept = 0, lty = 2) +
    stat_smooth(method = "lm") +
    ggtitle("Model predictions") +
    xlab("true value") +
    ylab("predicted value") +
    scale_color_d3() +
    scale_fill_d3() +
    facet_grid(dataset~model) +
    guides(fill=FALSE, color=FALSE) +
    theme_bw()

gg.residual <- ggplot(errors[errors$model %in% rownames(RMSE_summary.sorted),]) +
    aes(x = Ytrue, y = Ypred - Ytrue, color = model, fill = model) +
    geom_point(size = 2) +
    geom_abline(slope = 0, intercept = 0, lty = 2) +
    stat_smooth(method = "lm") +
    ggtitle("Residuals") +
    xlab("true value") +
    ylab("error (pred-true)") +
    scale_color_d3() +
    scale_fill_d3() +
    facet_grid(dataset~model) +
    #facet_grid(model~dataset) +
    guides(fill=FALSE, color=FALSE) +
    theme_bw()

grid.arrange(gg.error, gg.residual, nrow = 2)

# compute variable importance
viplist <- lapply(model.cv, function(x) varImp(x$model, useModel = T)$importance)
vip <- as.data.frame(lapply(viplist, function(this) this[colnames(training$X),1,drop=F]),
                     row.names = colnames(training$X))
colnames(vip) = names(model.cv)
vip[is.na(vip)] <- 0

# plot variable importance
gg.vip <- pheatmap(vip[,rownames(RMSE_summary.sorted)],
                   cluster_cols = F,
                   main = "variable importance")

gg.vip.transpose <- pheatmap(t(vip[,rownames(RMSE_summary.sorted)]),
                   cluster_rows  = F,
                   main = "variable importance")

save(model.cv,
     training,
     training.vif,
     testing,
     testing.raw,
     vip,
     errors,
     gg.vip,
     gg.vip.transpose,
     gg.error,
     gg.residual,
     gg.cv,
     RMSE_summary,
     RMSE_summary.sorted,
     RMSEMAE_summary,
     RMSEMAE_summary.sorted,
     RMSE.df,
     rValues,
     rValues.long,
     absolute_error.plot,
     file = "validation_models.RData")

