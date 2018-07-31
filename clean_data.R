
# function to clean the data
#
clean_data <- function(raw = read.csv("Table_of_features.csv"),
                       correlation_threshold = 1, # absolute Pearson correlation between features
                       imputation_threshold = 0.5, # max fraction of samples with NA before discarding feature
                       remove_linear_combinations = TRUE, # make full rank
                       outlier_threshold = 0.05, # outlier test p-value
                       cor.method = "pearson",
                       preProcess.method = c("center",
                                             "scale",
                                             "knnImpute",
                                             "YeoJohnson"),
                       impute.knn = 5,
                       vif_threshold = NULL,
                       vif_trace = FALSE){

    rownames(raw) <- raw$SUBID

    # which features are just scalings of another
    r <- cor(raw, use = "p", method = cor.method)
    graph <- graph_from_adjacency_matrix(abs(r) >= correlation_threshold, mode = "lower", diag = F)
    subgraph <- delete.vertices(simplify(graph), degree(graph) == 0)

    # pick one feature from each component as a representative
    # throw out the degenerates
    scc <- components(subgraph, mode = "strong") # strongly connected components
    representatives <- names(scc$membership)[match(unique(scc$membership), scc$membership)]
    degenerates <- setdiff(names(scc$membership), representatives)
    raw.nondegenerate <- raw[,!colnames(raw) %in% degenerates]
    if(length(degenerates)>0) cat(paste0("removed ", length(degenerates), " degenerate features \n"))

    # remove features with many NAs
    sumNA <- apply(raw.nondegenerate, 2, function(x) sum(is.na(x)))
    raw.good <- raw.nondegenerate[, sumNA <= imputation_threshold * nrow(raw.nondegenerate)]
    nremoved <- sum(sumNA > imputation_threshold * nrow(raw.nondegenerate))
    if(nremoved>0) cat(paste0("removed ", nremoved, " features due to missing data \n"))

    # impute the remaining missing data
    y.ix <- which(colnames(raw.good) == "Dependent_Feature")
    id.ix <- which(colnames(raw.good) == "SUBID")
    imputed.model <- preProcess(raw.good[,-c(id.ix, y.ix)],
                                method = preProcess.method,
                                outcome = raw.good[, y.ix],
                                k = impute.knn)
    imputed.x <- predict(imputed.model, raw.good[,-c(id.ix, y.ix)])
    imputed.y <- raw.good[, y.ix, drop = F]

    # remove columns that are linear combindations of another
    require(caret)
    if(remove_linear_combinations) lincombo.remove <- findLinearCombos(imputed.x)$remove
    if(!remove_linear_combinations) lincombo.remove <- NULL
    imputed.nocombo.x <- imputed.x[,!colnames(imputed.x) %in% colnames(imputed.x)[lincombo.remove]]
    imputed.nocombo.y <- imputed.y
    if(!is.null(lincombo.remove)) cat(paste0("removed ", length(lincombo.remove), " features as linear combinations \n"))

    # ordinary least squares regression
    require(car)
    ols.df <- as.data.frame(cbind(imputed.y, imputed.nocombo.x), row.names = raw.good[,id.ix])
    ols <- lm(Dependent_Feature ~ .,  data = ols.df)

    # remove outliers
    outliertest <- outlierTest(ols, cutoff = Inf, n.max = Inf, order = F) # Bonferonni p-value for most extreme obs
    outliertest$bonf.p[is.na(outliertest$bonf.p)] <- 1
    outlier <- outliertest$bonf.p < outlier_threshold
    imputed.nocombo.nooutlier.x <- imputed.nocombo.x[!outlier,]
    imputed.nocombo.nooutlier.y <- imputed.nocombo.y[!outlier,,drop = F]
    if(sum(outlier)>0) cat(paste0("removed ", sum(outlier), " outliers \n"))

    # other diagnostics
    heteroskedasticity <- ncvTest(ols) # non-constant variance test
    autocorrelation <- durbinWatsonTest(ols) # autocorrelation test

    if(!is.null(vif_threshold)){
        source("vif_func.R")
        vifs <- vif_func(in_frame = imputed.nocombo.nooutlier.x,
                         thresh = vif_threshold,
                         trace = vif_trace)
    } else {
        vifs <- colnames(imputed.nocombo.nooutlier.x)
    }

    imputed.nocombo.nooutlier.nocollinear.x <- imputed.nocombo.nooutlier.x[, vifs]
    imputed.nocombo.nooutlier.nocollinear.y <- imputed.nocombo.nooutlier.y
    nremoved <- ncol(imputed.nocombo.nooutlier.x)-length(vifs)
    if(nremoved>0) cat(paste0("removed ", nremoved, " features for large VIF \n"))

    # output
    cleaned <- list(X = imputed.nocombo.nooutlier.nocollinear.x,
                    Y = imputed.nocombo.nooutlier.nocollinear.y,
                    ols = ols,
                    r = r,
                    subgraph = subgraph,
                    outlier_test = outliertest,
                    skedastic = heteroskedasticity,
                    autocor = autocorrelation,
                    preProcess.model = imputed.model)

    return(cleaned)
}


