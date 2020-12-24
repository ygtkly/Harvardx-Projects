if(!require(RCurl)) install.packages("RCurl", repos = "http://cran.us.r-project.org")
if(!require(gdata)) install.packages("gdata", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(MVN)) install.packages("MVN", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")


library(ggplot2)
library(plyr)
library(tidyverse)
library(dplyr)
library(RCurl)
library(gdata)
library(matrixStats)
library(caret)
library(caretEnsemble)
library(knitr)
library(MVN)

#downloading data

url <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls'

mice.data <- read.xls(url)

#examining data

head(mice.data)

glimpse(mice.data)

summary(mice.data)

str(mice.data)

mice.data %>% group_by(class) %>% dplyr::summarize(n())

prop.table(table(mice.data$class))

#examining if there are NA's in the data

mice.data %>% dplyr::summarize(across(DYRK1A_N :CaNA_N, ~ sum(is.na(.x))))

na.cols <- mice.data %>% dplyr::summarize(across(DYRK1A_N :CaNA_N, ~ sum(is.na(.x))))

names(na.cols)[na.cols > 50]

#there are NA's in almost 1/3 of the instances, 
#and 5 predictors specifically have a large amount of NA's

#removing rows with NA's would cost us a lot of information, given the relatively small number of instances

#removing predictors with many NA's, and then using imputation would be logical,
#if the said predictors are not very useful anyway

#examining predictors with many NA's 

na.preds <- names(na.cols)[na.cols > 50]

mice.data %>% gather(protein, level, "DYRK1A_N" :"CaNA_N") %>%
  filter(protein %in% na.preds) %>%
  ggplot(aes(class, level)) + 
  geom_boxplot(aes(fill = class)) + 
  facet_wrap(protein~., scales = 'free_y')
  
#none of these predictors seem to distinguish well enough among classes
#so it makes sense to remove them
 
mice.data2 <- mice.data %>% select(-all_of(na.preds))

mice.data2 %>% dplyr::summarize(across(DYRK1A_N :CaNA_N, ~ sum(is.na(.x))))

#portioning predictors for more readable visualizations

pred1 <- names(mice.data2)[2:25]
pred2 <- colnames(mice.data2)[26:48]
pred3 <- colnames(mice.data2)[49:72]

#examining values by protein

mice.data2 %>% gather(protein, level, "DYRK1A_N" :"CaNA_N") %>%
  filter(protein %in% pred1) %>%
  ggplot(aes(protein, level)) + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle('Protein vs Level Group 1')

mice.data2 %>% gather(protein, level, "DYRK1A_N" :"CaNA_N") %>%
  filter(protein %in% pred2) %>%
  ggplot(aes(protein, level)) + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle('Protein vs Level Group 2')

mice.data2 %>% gather(protein, level, "DYRK1A_N" :"CaNA_N") %>%
  filter(protein %in% pred3) %>%
  ggplot(aes(protein, level)) + 
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  ggtitle('Protein vs Level Group 3')

# replacing NA's with median value of predictor, grouped by class
#median is chosen due to significant amount of points outside of boxplots 
#mean would be more influenced by these points

mice.data.final <- mice.data2 %>% group_by(class) %>%
  dplyr::mutate(across(DYRK1A_N: CaNA_N, ~ ifelse(is.na(.x), median(., na.rm = T), .x))) %>% ungroup() %>%
  as.data.frame()

#check to see if there are any NA's remaining

mice.data.final %>% dplyr::summarize(across(DYRK1A_N :CaNA_N, ~ sum(is.na(.x))))

#creating training and test set

set.seed(1986, sample.kind = 'Rounding')

y <- mice.data.final$class

test.index <- createDataPartition(y, times = 1, p = 0.4, list = F)

test.set <- mice.data.final[test.index,]

training.set <- mice.data.final[-test.index,]

test.set.preds <- test.set %>% 
  select(-Behavior, -Genotype, -Treatment, -class, -MouseID) %>% as.matrix()

training.set.preds <- training.set %>% 
  select(-Behavior, -Genotype, -Treatment, -class, -MouseID) %>% as.matrix()


#checking distribution of classes in training and test sets

training.set %>% group_by(class) %>% dplyr::summarize(n())

test.set %>% group_by(class) %>% dplyr::summarize(n())

#the distribution of classes is more or less balanced


#calculating standard deviations and variance of predictors
sds <- colSds(training.set.preds, na.rm = T)
variances <- colVars(training.set.preds, na.rm = T)

min(sds)
max(sds)

min(variances)
max(variances)

#standardization seems like a good idea before we implement PCA since the sd and variance among
#different predictors have high variability 


#checking if there are predictors with almost zero variance
nzv <- nearZeroVar(training.set.preds)
nzv

#no predictors have near zero variance. there is no chance of elimination predictors with this method

#create correlation matrix 

cor.mat <- cor(training.set.preds)

#make diagonal and lower triangle 0 to avoid overcalculating proportion correlations

diag(cor.mat) <- 0

cor.mat[lower.tri(cor.mat)] <- 0

# calculate proportion of different levels of correlation

very.high.correlation.per <- (sum(cor.mat > 0.9 | cor.mat < -0.9))/(71*71)
names(very.high.correlation.per) <- 'Very High Correlation %'
high.correlation.per <- (sum(cor.mat > 0.7 | cor.mat < -0.7))/(71*71)
names(high.correlation.per) <- 'High Correlation %'
moderate.correlation.per <- (sum(cor.mat > 0.5 | cor.mat < -0.5))/(71*71)
names(moderate.correlation.per) <- 'Moderate Correlation %'

knitr::kable(c(very.high.correlation.per, high.correlation.per, moderate.correlation.per))

#multicollinearity doesn't seem to be a big problem since proportions are quite low

#check for multivariate normality

mvn.test.dh <- mvn(training.set.preds, mvnTest = 'dh')
mvn.test.hz <- mvn(training.set.preds, mvnTest = 'hz')
mvn.test.dh$multivariateNormality
mvn.test.hz$multivariateNormality

#gathering the predictors for analysis
proteins.gather <- training.set %>% gather(protein, level, "DYRK1A_N" :"CaNA_N")


#creating box plots to see if there are predictors that distinguish between genotype

proteins.gather %>% filter(protein %in% pred1) %>% 
  ggplot(aes(Genotype, level)) + 
  geom_boxplot(aes(fill = Genotype)) + 
  facet_wrap(protein~., scales = 'free_y') + 
  ggtitle('Examination of Genotype Distinguishing Predictors Part 1')

#result BRAF_N , pELK_N and p_ERK_N seem like they could be partially useful in group 1

proteins.gather %>% filter(protein %in% pred2) %>% 
  ggplot(aes(Genotype, level)) + 
  geom_boxplot(aes(fill = Genotype)) + 
  facet_wrap(protein~., scales = 'free_y') + 
  ggtitle('Examination of Genotype Distinguishing Predictors Part 2')

#result - APP_N, DSCR1_N and MTOR_N, PMTOR_N seems like they could be partially useful in group 2

proteins.gather %>% filter(protein %in% pred3) %>% 
  ggplot(aes(Genotype, level)) + 
  geom_boxplot(aes(fill = Genotype)) + 
  facet_wrap(protein~., scales = 'free_y') + 
  ggtitle('Examination of Genotype Distinguishing Predictors Part 3')

# result - no useful predictors in group 3


#creating point plots to see if there are predictors that distinguish between treatment type

proteins.gather %>% filter(protein %in% pred1) %>% 
  ggplot(aes(Treatment, level)) + 
  geom_boxplot(aes(fill = Treatment)) +
  facet_wrap(protein~., scales = 'free_y') +
  ggtitle('Examination of Treatment Distinguishing Predictors Part 1')

#result BRAF_N , pELK_N and p_ERK_N seem like they could be partially useful in group 1

proteins.gather %>% filter(protein %in% pred2) %>% 
  ggplot(aes(Treatment, level)) + 
  geom_boxplot(aes(fill = Treatment)) +
  facet_wrap(protein~., scales = 'free_y') +
  ggtitle('Examination of Treatment Distinguishing Predictors Part 2')

# result - pMTOR_N and pPKCG_N seem like they could be partially useful group 2

proteins.gather %>% filter(protein %in% pred3) %>% 
  ggplot(aes(Treatment, level)) + 
  geom_boxplot(aes(fill = Treatment)) +
  facet_wrap(protein~., scales = 'free_y') + 
  ggtitle('Examination of Treatment Distinguishing Predictors Part 3')

#result -  no useful predictors in group 3

#creating point plots to see if there are predictors that distinguish between behaviour type

proteins.gather %>% filter(protein %in% pred1) %>% 
  ggplot(aes(Behavior, level)) + 
  geom_boxplot(aes(fill = Behavior)) +
  facet_wrap(.~protein, scales = 'free_y') +
  ggtitle('Examination of Behavior Distinguishing Predictors Part 1')

#result pERK_N, BRAF_N, DYRK1A_N best distinguieshes behavior in group 1

proteins.gather %>% filter(protein %in% pred2) %>% 
  ggplot(aes(Behavior, level)) + 
  geom_boxplot(aes(fill = Behavior)) +
  facet_wrap(.~protein, scales = 'free_y') +
  ggtitle('Examination of Behavior Distinguishing Predictors Part 2')

#result - SOD1_N distinguieshes behavior in group 2

proteins.gather %>% filter(protein %in% pred3) %>% 
  ggplot(aes(Behavior, level)) + 
  geom_boxplot(aes(fill = Behavior)) +
  facet_wrap(.~protein, scales = 'free_y') +
  ggtitle('Examination of Behavior Distinguishing Predictors Part 3')

#result CaNA_N best distinguieshes behavior in group 3


#MODEL 1: GUESSING

set.seed(1986, sample.kind = 'Rounding')

guess.predicted <- sample(levels(test.set$class), length(test.index), replace = T) %>% 
  factor(levels = levels(test.set$class))

accuracy.guess <- mean(guess.predicted == test.set$class)

#MODEL 2: CLASSIFICATION TREE WITH MANUALLY CHOSEN PREDICTOR SELECTION

#below predictors are chosen by eye from the graphs produced, anr will be used for model 2-3-4.

chosen.predictors <- c('BRAF_N', 'pELK_N', 'pERK_N', 'APP_N', 'DSCR1_N', 'MTOR_N', 'pMTOR_N', 
                       'DYRK1A_N', 'SOD1_N', 'CaNA_N')

chosen.index <- which(colnames(training.set.preds) %in% chosen.predictors)

training.chosen.predictors <- training.set.preds[,chosen.index]

test.chosen.predictors <- test.set.preds[,chosen.index]

set.seed(1986, sample.kind = 'Rounding')

fit.manual.rpart <- train(training.chosen.predictors, training.set$class, 
                      method = 'rpart', 
                      tuneGrid = data.frame(cp = seq(0.0, 0.05, len = 40)))

accuracy.manual.rpart <- mean(predict(fit.manual.rpart, test.chosen.predictors) == test.set$class)


#MODEL 3:  K-NEAREST NEIGHBORS WITH MANUAL PREDICTOR SELECTION

set.seed(1986, sample.kind = 'Rounding')

fit.manual.knn <- train(training.chosen.predictors, training.set$class, 
                 method = 'knn', 
                 tuneGrid = data.frame(k = seq(2, 20, 1)))

accuracy.manual.knn <- mean(predict(fit.manual.knn, test.chosen.predictors) == test.set$class)

#MODEL 4: RANDOM FOREST WITH MANUAL PREDICTOR SELECTION

set.seed(1986, sample.kind = 'Rounding')

fit.manual.rf <-train(training.chosen.predictors, training.set$class,
               method = 'rf', 
               tuneGrid = data.frame(mtry = seq(1, 8, 1)), importance = T, ntree = 500)

accuracy.manual.rf <- mean(predict(fit.manual.rf, test.chosen.predictors) == test.set$class)

fit.manual.rf$bestTune

imp.manual.rf <- varImp(fit.manual.rf)

imp.pred.manual.rf <- rownames(imp.manual.rf$importance)

#MODEL 5: CLASSIFICATION TREE WITH ALL PREDICTORS

set.seed(1986, sample.kind = 'Rounding')

fit.rpart <- train(training.set.preds, training.set$class, 
                          method = 'rpart', 
                          tuneGrid = data.frame(cp = seq(0.0, 0.05, len = 40)))

accuracy.rpart <- mean(predict(fit.rpart, test.set.preds) == test.set$class)

#MODEL 6:  K-NEAREST NEIGHBORS WITH ALL PREDICTORS

set.seed(1986, sample.kind = 'Rounding')

fit.knn <- train(training.set.preds, training.set$class, 
                        method = 'knn', 
                        tuneGrid = data.frame(k = seq(2, 20, 1)))

accuracy.knn <- mean(predict(fit.knn, test.set.preds) == test.set$class)

#MODEL 7: RANDOM FOREST WITH ALL PREDICTORS

set.seed(1986, sample.kind = 'Rounding')

fit.rf <-train(training.set.preds, training.set$class,
                      method = 'rf', 
                      tuneGrid = data.frame(mtry = seq(1, 8, 1)), importance = T, ntree = 500)

accuracy.rf <- mean(predict(fit.rf, test.set.preds) == test.set$class)

fit.rf$bestTune

imp.rf <- varImp(fit.rf)


#MODEL 8 CLASSIFICATION TREE WITH PCA

#PCA FOR MODELS 8-9-10

#scaling training and test sets

standard.training.pred <- data.frame(scale(training.set.preds))

standard.test.pred <- sweep(test.set.preds, 2, colMeans(training.set.preds))
standard.test.pred <- sweep(standard.test.pred, 2, colSds(training.set.preds), FUN = '/')

#computing pca
pca <- prcomp(standard.training.pred)

summary(pca)

var.explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
plot(var.explained)
abline(h = 1, col = 'red')

# %100 variability is explained at around 45th principal component

set.seed(1986, sample.kind = 'Rounding')

pcs <- data.frame(pca$x, class = training.set$class)

pcs %>% ggplot(aes(PC1, PC2, color = class)) + geom_point()
pcs %>% ggplot(aes(PC2, PC3, color = class)) + geom_point()
pcs %>% ggplot(aes(PC3, PC4, color = class)) + geom_point()

#tuning princical components to be included for rpart

pcas.included <- seq(2,71,1)


rpart.tune.pc <- function(pcn) {
  fit.pca.rpart <- train(pca$x[, 1:pcn], training.set$class,
                         method = 'rpart',
                         tuneGrid = data.frame(cp = seq(0, 0.05, length = 40)))
  return(caretEnsemble:: getMetric(fit.pca.rpart))
}

set.seed(1986, sample.kind = 'Rounding')

rpart.pc.options <- sapply(X = pcas.included , FUN =  rpart.tune.pc)

rpart.best.pc <- pcas.included[which.max(rpart.pc.options)]


set.seed(1986, sample.kind = 'Rounding')

fit.pca.rpart <- train(pca$x[, 1:rpart.best.pc], training.set$class,
                       method = 'rpart',
                       tuneGrid = data.frame(cp = seq(0, 0.05, length = 20)))

accuracy.pca.rpart <- mean(predict(fit.pca.rpart, predict(pca, standard.test.pred)) == test.set$class)

#MODEL 9 K-NEAREST NEIGHBORS WITH PCA

knn.tune.pc <- function(pcn) {
  fit.pca.knn <- train(pca$x[, 1:pcn], training.set$class,
                         method = 'knn',
                         tuneGrid = data.frame(k = seq(2, 20, 2)))
  return(caretEnsemble:: getMetric(fit.pca.knn))
}


set.seed(1986, sample.kind = 'Rounding')

knn.pc.options <- sapply(X = pcas.included , FUN =  knn.tune.pc)

knn.best.pc <- pcas.included[which.max(knn.pc.options)]

set.seed(1986, sample.kind = 'Rounding')

fit.pca.knn <- train(pca$x[, 1:knn.best.pc], training.set$class,
                       method = 'knn',
                       tuneGrid = data.frame(k = seq(2, 20, 2)))


accuracy.pca.knn <- mean(predict(fit.pca.knn, predict(pca, standard.test.pred)) == test.set$class)

#MODEL 10 RANDOM FOREST WITH PCA


rf.tune.pc <- function(pcn) {
  fit.pca.rf <- train(pca$x[, 1:pcn], training.set$class,
                         method = 'rf',
                         tuneGrid = data.frame(mtry = seq(1, 8, 1)), ntree = 100)
  return(caretEnsemble:: getMetric(fit.pca.rf))
}


set.seed(1986, sample.kind = 'Rounding')

rf.pc.options <- sapply(X = pcas.included , FUN =  rf.tune.pc)

rf.best.pc <- pcas.included[which.max(rf.pc.options)]

set.seed(1986, sample.kind = 'Rounding')

fit.pca.rf <- train(pca$x[, 1:rf.best.pc], training.set$class,
                     method = 'rf',
                     tuneGrid = data.frame(mtry = seq(1, 8, 1)), importance = T, ntree = 500)


accuracy.pca.rf <- mean(predict(fit.pca.rf, predict(pca, standard.test.pred)) == test.set$class)

# ACCURACY BY MODEL TABLE

accuracy.by.model <- data.frame(model = c('Guessing Model', 
                             'Manual Cls. Tree Model', 
                             'Manual KNN Model', 
                             'Manual Random Forest Model', 
                             'Regular Cls. Tree Model', 
                             'Regular KNN Model',
                             'Regular Random Forest Model', 
                             'PCA Cls. Tree Model', 
                             'PCA KNN Model', 
                             'PCA Random Forest Model'))

accuracy.by.model <- accuracy.by.model %>% mutate(accuracy =  c(accuracy.guess, 
                                              accuracy.manual.rpart,
                                              accuracy.manual.knn, 
                                              accuracy.manual.rf, 
                                              accuracy.rpart, 
                                              accuracy.knn, 
                                              accuracy.rf, 
                                              accuracy.pca.rpart, 
                                              accuracy.pca.knn, 
                                              accuracy.pca.rf))


accuracy.by.model <- accuracy.by.model %>% arrange(desc(accuracy)) 

accuracy.by.model

#FINAL MODEL - ENSEMBLE BY HIGHEST VOTE OF 3 MOST SUCCESSFUL MODELS

fit.ensemble <- data.frame(regular.rf = predict(fit.rf, test.set.preds),
                           pca.rf = predict(fit.pca.rf, predict(pca, standard.test.pred)),
                           pca.knn = predict(fit.pca.knn, predict(pca, standard.test.pred)))


fit.ensemble <- fit.ensemble %>% mutate(ensemble.predicted = ifelse(regular.rf == pca.rf & regular.rf == pca.knn, as.character(regular.rf),
                                                                    ifelse(regular.rf == pca.rf, as.character(regular.rf), 
                                                                           ifelse(regular.rf == pca.knn, as.character(regular.rf), 
                                                                                  as.character(pca.rf))))) %>%
  mutate(ensemble.predicted = as.factor(ensemble.predicted)) %>%
  mutate(test.y = test.set$class)

#RESULT

accuracy.ensemble = mean(fit.ensemble$ensemble.predicted == fit.ensemble$test.y)

accuracy.ensemble


