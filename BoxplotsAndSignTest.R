library(ggplot2)

loadFile <- function(fileName, enemy, mutation){
  df <-data.frame(read.table(fileName))
  colnames(df) <- c('IndividualGain')
  df$enemy <- enemy
  df$mutation <- as.factor(mutation)
  return(df)
}

enem1uniform<- loadFile("individual_gain_enemy1_mutuniform",1,'uniform')
enem1norm <- loadFile("individual_gain_enemy1_mutnormal",1,'normal')

enem2uniform<- loadFile("individual_gain_enemy2_mutuniform",2,'uniform')
enem2norm <- loadFile("individual_gain_enemy2_mutnormal",2,'normal')

enem3uniform<- loadFile("individual_gain_enemy3_mutuniform",3,'uniform')

enem3norm <- loadFile("individual_gain_enemy3_mutnormal",3,'normal')

combined <- rbind(rbind(rbind(rbind(rbind(enem1norm,enem1uniform),enem3norm),enem3uniform),enem2norm),enem2uniform)
combined$enemy <- as.factor(combined$enemy)
combined$mutation<- as.factor(combined$mutation)
ggplot(combined,aes(x =enemy, y = IndividualGain,fill =mutation)) + geom_boxplot()

#Wilcox test
combinedEnemy1 = rbind(enem1uniform,enem1norm)
combinedEnemy2 <- rbind(enem2uniform,enem2norm)
combinedEnemy3 <- rbind(enem3norm,enem3uniform)
res1 <- wilcox.test(IndividualGain ~ mutation, data = combinedEnemy1, exact = FALSE)
res1
res2 <- wilcox.test(IndividualGain ~ mutation, data = combinedEnemy2, exact = FALSE)
res2
res3 <- wilcox.test(IndividualGain ~ mutation, data = combinedEnemy3, exact = FALSE)
res3
