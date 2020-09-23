library(ggplot2)

loadFile <- function(fileName, enemy, mutation){
  df <-data.frame(read.table(fileName))
  colnames(df) <- c('Individual Gain')
  df$enemy <- enemy
  df$mutation <- mutation 
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
ggplot(combined,aes(x =enemy, y = `Individual Gain`,fill =mutation)) + geom_boxplot()

