require(pheatmap)
require(ggbiplot)
require(ggplot2)
require(ggsci)
require(ggpubr)
require(lmerTest)
require(gbm)
require(reshape2)
require(stringr)
require(igraph)
require(factoextra)

# partition samples 80%/20% based on 5 quantiles of y
source("data_partitions.R")

# load function to clean the training data
source("clean_data.R")

# get the full data set
full <- clean_data(raw = read.csv("full.csv"),
                   correlation_threshold = Inf,
                   remove_linear_combinations = F)

full.vif <- clean_data(raw = read.csv("full.csv"),
                   correlation_threshold = .975,
                   vif_threshold = 10,
                   remove_linear_combinations = F)

# look at clustering
doublestandardize <- function(x, iter = 20){
    for(ii in 1:iter) x <- t(scale(t(scale(x))))
    return(x)
}
full$Xds <- doublestandardize(full$X)
gg.clustering <- pheatmap(full$Xds,
         border_color = NA,
         fontsize_col = 5,
         fontsize_row = 5,
         cutree_rows = 3,
         cutree_cols = 4,
         main = "Clustering cleaned data")


# principal component analysis
pca <- prcomp(full$X, center = T, scale. = T)
eig <- cbind(Dependent_Feature = full$Y, as.data.frame(pca$x))
pc100 <- sum(cumsum(pca$sdev^2)/sum(pca$sdev^2) < 1)
pcr <- lm(Dependent_Feature ~ ., data = eig[,1:(pc100+2)])
sumtable.pcr <- summary(pcr)
top2pcs <- sort(sumtable.pcr$coefficients[-1,4], index.return = T)$ix[1:2]
gg.biplot <- ggbiplot(pca,
                      choices = top2pcs,
                      labels = rownames(full$X),
                      groups = cut(full$Y[[1]],
                                   quantile(full$Y[[1]],
                                            probs = seq(0, 1, length.out = 5)),
                                   ordered_result = T,
                                   include.lowest = T),
                      ellipse = T) +
    theme_classic() +
    scale_color_d3()
print(gg.biplot)

# a better look at the features
gg.pcavarplot <- fviz_pca_var(pca,
                              axes = top2pcs,
                              col.var = "contrib",
                              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                              repel = TRUE, # Avoid text overlapping
                              alpha.var = 0.5,
                              select.var = list(contrib = 40)
)
print(gg.pcavarplot)


# Y correlation to features
cortest <- apply(full$X, 2, function(x) cor.test(x, full$Y[[1]]))
rho <- data.frame("correlation" = sapply(cortest, function(x) x$estimate), row.names = names(cortest))
rho$fwer <- p.adjust(sapply(cortest, function(x) x$p.value), method = "bonferroni")
rho$feature <- rownames(rho)
rho$feature <- factor(rho$feature, levels = rho$feature[order(rho$correlation)])
gg.correlation <- ggplot(rho[abs(rho$fwer) < 0.05,]) +
    aes(x = feature, y = correlation) +
    geom_bar(stat="identity", aes(fill = correlation)) +
    scale_fill_continuous(guide = F) +
    ggtitle("Significant features (FWER=5%)") +
    theme_classic() +
    ylab("Pearson correlation") +
    coord_flip()
print(gg.correlation)

save(gg.biplot,
     gg.pcavarplot,
     gg.correlation,
     gg.clustering,
     rho,
     full,
     full.vif,
     file = "pca_cor_clust.RData")

