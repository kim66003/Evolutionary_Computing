library(ggplot2)

  loadFile <- function(group, sharing){
  charVector <- c('individual_gain_',sharing,'_',group)
  fileName <- paste(charVector,collapse = "")
  df <-data.frame(read.table(fileName))
  colnames(df) <- c('IndividualGain')
  df$group <- as.factor(group)
  df$sharing <- as.factor(sharing)
  return(df)
}

sharing456 <- loadFile('[4, 5, 6]','fitness_sharing')
sharing478 <- loadFile('[4, 7, 8]','fitness_sharing')
noSharing456<- loadFile('[4, 5, 6]','no_fitness_sharing')
noSharing478 <- loadFile('[4, 7, 8]','no_fitness_sharing')


combined <- rbind(rbind(rbind(sharing456,sharing478),noSharing456),noSharing478)
ggplot(combined,aes(x =group, y = IndividualGain,fill =sharing)) + geom_boxplot() + labs(y="Mean Individual Gain", x = "Enemy Groups used in training")

#Wilcox tests
combined456 = rbind(sharing456,noSharing456)
testStuff <- rbind(combined456,combined456)
combined478 = rbind(sharing478,noSharing478)

sharingRes1 <- wilcox.test(IndividualGain ~ sharing, data = combined456, exact = FALSE)
sharingRes2 <- wilcox.test(IndividualGain ~ sharing, data = combined478, exact = FALSE)

combinedSharing <- rbind(sharing456,sharing478)
combinedNoSharing <- rbind(noSharing456,noSharing478)
groupRes1 <- wilcox.test(IndividualGain ~ group, data = combinedSharing, exact = FALSE)

groupRes2 <- wilcox.test(IndividualGain ~ group, data = combinedNoSharing, exact = FALSE)
groupRes1
groupRes2
sharingRes1
sharingRes2