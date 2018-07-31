#________________________________________________________________________
# partition samples 80%/20% based on 5 quantiles of y

set.seed(141)

# read data file and partition
raw = read.csv("Table_of_features.csv")

require(caret)
inTraining <- createDataPartition(raw$Dependent_Feature,
                                  p = .8,
                                  groups = 5,
                                  list = F)

# write out to files
write.csv(raw, file = "full.csv", row.names = F)
write.csv(raw[inTraining,], file = "training.csv", row.names = F)
write.csv(raw[-inTraining,], file = "testing.csv", row.names = F)
rm(raw)

#_________________________________________________________
