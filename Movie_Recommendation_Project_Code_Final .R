if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(stringr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#-------------------------------end of the provided code----------------------------------------------


#creating test and training sets


y <- edx$rating

test.index <- createDataPartition(y, times = 1, p = 0.2, list = F)

test.set <- edx[test.index, ]
training.set <- edx[-test.index,]

#removing users and movies that do not appear in the training set from the test set

test.set <- test.set %>%
  semi_join(training.set, by = 'movieId') %>%
  semi_join(training.set, by = 'userId')


#data exploration

summary(edx)
summary(training.set)

training.set %>% summarize(movies =  n_distinct(movieId), users = n_distinct(userId), genres = n_distinct(genres))

#examining the movie effect

training.set %>% filter(n() > 100) %>% group_by(movieId) %>% summarize(avg = mean(rating)) %>%
  ggplot() + geom_histogram(aes(avg), color = 'blue')

#examining the user effect

training.set %>% filter(n() > 80) %>% group_by(userId) %>% summarize(avg = mean(rating)) %>%
  ggplot() + geom_histogram(aes(avg), color = 'red')

# examining the genre effect among some major genres

training.set %>% filter(genres %in% c('Drama', 'Comedy', 'Horror', 'Sci-Fi', 'Thriller')) %>% 
  group_by(genres) %>% 
  ggplot(aes(x = genres, y = rating, color = genres)) + geom_boxplot()

training.set %>% filter(genres %in% c('Drama', 'Comedy', 'Horror', 'Sci-Fi', 'Thriller')) %>% 
  group_by(genres) %>% summarize(avg = mean(rating)) %>%
  ggplot(aes(x = genres, y = avg, color = genres)) + geom_point()

training.set %>% filter(genres %in% c('Drama', 'Comedy', 'Horror', 'Sci-Fi', 'Thriller')) %>% 
  group_by(genres) %>% summarize(avg.rating = mean(rating)) %>% knitr::kable()


#examining the year effect

year.train <- training.set %>% mutate(year = str_extract(title, pattern =  "\\(\\d{4}\\)")) %>% 
  mutate(year = str_remove(year, '\\(')) %>% 
  mutate(year = str_remove(year, '\\)')) %>%
  mutate(year = as.numeric(year))%>%
  select(movieId,year) %>% distinct()

training.set %>% left_join(year.train, by = 'movieId') %>%
  group_by(year) %>%
  summarize(rt.by.year = mean(rating)) %>% summarize(mean.avg.years  = mean(rt.by.year),
                                                    sd.avg.years = sd(rt.by.year),
                                                    min.avg.years =  min(rt.by.year), 
                                                    median.avg.years = median(rt.by.year), 
                                                    max.avg.years = max(rt.by.year)) %>%
  knitr::kable()

year.edx <- edx %>% mutate(year = str_extract(title, pattern =  "\\(\\d{4}\\)")) %>% 
  mutate(year = str_remove(year, '\\(')) %>% 
  mutate(year = str_remove(year, '\\)')) %>%
  mutate(year = as.numeric(year))%>%
  select(movieId,year) %>% distinct()

edx.with.year <- edx %>% left_join(year.edx, by='movieId') %>% mutate(year = as.factor(year))


aov.test <- aov(rating ~ year, data = edx.with.year)
summary(aov.test)


#MODEL CREATION

#calculating edx set overall average rating

mu.edx <- mean(edx$rating)

#calculating training set overall average rating

mu.train <- mean(training.set$rating)

#calculating overall average model rmse for baseline

overall.mean.model.rmse <- sqrt(mean((test.set$rating - mu.train)^2))

#creating k-fold validation sets


indexes <- createFolds(training.set$rating, k = 10)

#tuning regularized movie effect

movie.effect.tuner <- seq(1, 3, 0.1)

movie.tuner.rmses <- lapply(indexes, function(fold){
  rmses <- sapply(movie.effect.tuner, function(lambda){
    mu <- mean(training.set[-fold,]$rating)
    summer <- training.set[-fold,] %>% 
      group_by(movieId) %>%
      summarize(sum = sum(rating - mu), rated = n())
    predicted <- training.set[fold,] %>% 
      semi_join(training.set[-fold,], by = 'movieId') %>%
      left_join(summer, by = 'movieId') %>%
      mutate(me = sum/(rated + lambda)) %>%
      mutate(pred = mu + me) %>%
      summarize(rmse = sqrt(mean((rating - pred)^2)))
  })
  return(movie.effect.tuner[which.min(rmses)])
}) 

#determining the best penalty term
best.movie.tuner <- mean(as.numeric(movie.tuner.rmses))

#calculating movie effect for the whole training set & test set

tr.movie.effect <- training.set %>%
  group_by(movieId) %>%
  summarize(me = sum(rating - mu.train)/(n() + best.movie.tuner))

edx.movie.effect <- edx %>%
  group_by(movieId) %>%
  summarize(me = sum(rating - mu.train)/(n() + best.movie.tuner))

#calculating rmse on the test set

movie.model.rmse <- test.set %>%
  left_join(tr.movie.effect, by = 'movieId') %>%
  mutate(predicted = mu.train + me) %>%
  summarize(rmse.movie = sqrt(mean((rating - predicted)^2)))


#tuning regularized user effect

user.effect.tuner <- seq(3.5, 6.5, 0.1)

user.tuner.rmses <- lapply(indexes, function(fold){
  rmses <- sapply(user.effect.tuner, function(lambda){
    mu <- mean(training.set[-fold,]$rating)
    summer <- training.set[-fold,] %>% 
      left_join(tr.movie.effect, by = 'movieId') %>%
      group_by(userId) %>%
      summarize(sum = sum(rating - mu - me), rated = n())
    predicted <- training.set[fold,] %>% 
      semi_join(training.set[-fold,], by = 'movieId') %>%
      left_join(tr.movie.effect, by = 'movieId') %>%
      left_join(summer, by = 'userId') %>%
      mutate(ue = sum/(rated + lambda)) %>%
      mutate(pred = mu + me + ue) %>%
      summarize(rmse = sqrt(mean((rating - pred)^2)))
  })
  print(rmses)
  return(user.effect.tuner[which.min(rmses)])
}) 

#determining the best penalty term
best.user.tuner <- mean(as.numeric(user.tuner.rmses))

#calculating user effect for the whole training set & test set

tr.user.effect <- training.set %>%
  left_join(tr.movie.effect, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(ue = sum(rating - mu.train - me)/(n() + best.user.tuner))

edx.user.effect <- edx %>%
  left_join(edx.movie.effect, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(ue = sum(rating - mu.edx - me)/(n() + best.user.tuner))

# calculating rmse on the test set

movie.user.model.rmse <- test.set %>%
  left_join(tr.movie.effect, by = 'movieId') %>%
  left_join(tr.user.effect, by = 'userId') %>%
  mutate(predicted = mu.train + me + ue) %>%
  summarize(rmse.movie.user = sqrt(mean((rating - predicted)^2)))


#tuning regluarized genre effect

genre.effect.tuner <- seq(500, 700, 10)

genre.tuner.rmses <- lapply(indexes, function(fold){
  rmses <- sapply(genre.effect.tuner, function(lambda){
    mu <- mean(training.set[-fold,]$rating)
    summer <- training.set[-fold,] %>% 
      left_join(tr.movie.effect, by = 'movieId') %>%
      left_join(tr.user.effect, by = 'userId') %>%
      group_by(genres) %>%
      summarize(sum = sum(rating - mu - me - ue), rated = n())
    predicted <- training.set[fold,] %>% 
      semi_join(training.set[-fold,], by = 'movieId') %>%
      left_join(tr.movie.effect, by = 'movieId') %>%
      left_join(tr.user.effect, by = 'userId') %>%
      left_join(summer, by = 'genres') %>%
      mutate(ge = sum/(rated + lambda)) %>%
      mutate(pred = mu + me + ue + ge) %>%
      summarize(rmse = sqrt(mean((rating - pred)^2)))
  })
  print(rmses)
  return(user.effect.tuner[which.min(rmses)])
}) 

#determining the best penalty term
best.genre.tuner <- mean(as.numeric(genre.tuner.rmses))

#calculating genre effect for the whole training set & test set

tr.genre.effect <- training.set %>%
  left_join(tr.movie.effect, by = 'movieId') %>%
  left_join(tr.user.effect, by = 'userId') %>%
  group_by(genres) %>%
  summarize(ge = sum(rating - mu.train - me - ue)/(n() + best.genre.tuner))

edx.genre.effect <- edx %>%
  left_join(edx.movie.effect, by = 'movieId') %>%
  left_join(edx.user.effect, by = 'userId') %>%
  group_by(genres) %>%
  summarize(ge = sum(rating - mu.edx - me - ue)/(n() + best.genre.tuner))


# calculating rmse on the test set

movie.user.genre.model.rmse <- test.set %>%
  left_join(tr.movie.effect, by = 'movieId') %>%
  left_join(tr.user.effect, by = 'userId') %>%
  left_join(tr.genre.effect, by = 'genres') %>%
  mutate(predicted = mu.train + me + ue + ge) %>%
  summarize(rmse.movie.user.genre = sqrt(mean((rating - predicted)^2)))

#extracting year from title for training set

year.train <- training.set %>% mutate(year = str_extract(title, pattern =  "\\(\\d{4}\\)")) %>% 
  mutate(year = str_remove(year, '\\(')) %>% 
  mutate(year = str_remove(year, '\\)')) %>%
  mutate(year = as.numeric(year))%>%
  select(movieId,year) %>% distinct()

#extracting year from title for test set

year.test <- test.set %>% mutate(year = str_extract(title, pattern =  "\\(\\d{4}\\)")) %>% 
  mutate(year = str_remove(year, '\\(')) %>% 
  mutate(year = str_remove(year, '\\)')) %>%
  mutate(year = as.numeric(year))%>%
  select(movieId,year) %>% distinct()

#extracting year from title for edx

year.edx <- edx %>% mutate(year = str_extract(title, pattern =  "\\(\\d{4}\\)")) %>% 
  mutate(year = str_remove(year, '\\(')) %>% 
  mutate(year = str_remove(year, '\\)')) %>%
  mutate(year = as.numeric(year))%>%
  select(movieId,year) %>% distinct()

#tuning regluarized year effect

year.effect.tuner <- seq(0, 20, 1)

year.tuner.rmses <- lapply(indexes, function(fold){
  rmses <- sapply(year.effect.tuner, function(lambda){
    mu <- mean(training.set[-fold,]$rating)
    summer <- training.set[-fold,] %>% 
      left_join(tr.movie.effect, by = 'movieId') %>%
      left_join(tr.user.effect, by = 'userId') %>%
      left_join(tr.genre.effect, by = 'genres') %>%
      left_join(as.data.frame(year.train), by = 'movieId') %>%
      group_by(year) %>%
      summarize(sum = sum(rating - mu - me - ue - ge), rated = n())
    predicted <- training.set[fold,] %>% 
      semi_join(training.set[-fold,], by = 'movieId') %>%
      left_join(tr.movie.effect, by = 'movieId') %>%
      left_join(tr.user.effect, by = 'userId') %>%
      left_join(tr.genre.effect, by = 'genres') %>%
      left_join(as.data.frame(year.train), by = 'movieId') %>%
      left_join(summer, by = 'year') %>%
      mutate(ye = sum/(rated + lambda)) %>%
      mutate(pred = mu + me + ue + ge+ ye) %>%
      summarize(rmse = sqrt(mean((rating - pred)^2)))
  })
  print(rmses)
  return(user.effect.tuner[which.min(rmses)])
}) 

#determining the best penalty term
best.year.tuner <- mean(as.numeric(year.tuner.rmses))

#calculating genre effect for the whole training set & test set

tr.year.effect <- training.set %>%
  left_join(tr.movie.effect, by = 'movieId') %>%
  left_join(tr.user.effect, by = 'userId') %>%
  left_join(tr.genre.effect, by = 'genres') %>%
  left_join(year.train, by = 'movieId') %>%
  group_by(year) %>%
  summarize(ye = sum(rating - mu.train - me - ue - ge)/(n() + best.year.tuner))

edx.year.effect <- edx %>%
  left_join(edx.movie.effect, by = 'movieId') %>%
  left_join(edx.user.effect, by = 'userId') %>%
  left_join(edx.genre.effect, by = 'genres') %>%
  left_join(year.edx, by = 'movieId') %>%
  group_by(year) %>%
  summarize(ye = sum(rating - mu.edx - me - ue - ge)/(n() + best.year.tuner))



# calculating rmse on the test set

movie.user.genre.year.model.rmse <- test.set %>%
  left_join(tr.movie.effect, by = 'movieId') %>%
  left_join(tr.user.effect, by = 'userId') %>%
  left_join(tr.genre.effect, by = 'genres') %>%
  left_join(year.test, by = 'movieId') %>%
  left_join(tr.year.effect, by = 'year') %>%
  mutate(predicted = mu.train + me + ue + ge + ye) %>%
  summarize(rmse.movie.user.genre.year = sqrt(mean((rating - predicted)^2)))

# rmse table by model 

rmse.by.model <- data.frame(rmse = c(overall.mean.model.rmse,
                                     movie.model.rmse, 
                                     movie.user.model.rmse, 
                                     movie.user.genre.model.rmse, 
                                     movie.user.genre.year.model.rmse))

rmse.by.model <- rmse.by.model %>% matrix(nrow = 5)
colnames(rmse.by.model) <- 'RMSE'
rownames(rmse.by.model) <- c('Overall Mean Model', 
                             'Reg. Movie Effect Model', 
                             'Reg. Movie + User Effect Model', 
                             'Reg. Movie + User + Genre Effect Model', 
                             'Reg. Movie + User + Genre + Year Effect Model')

rmse.by.model <- knitr::kable(rmse.by.model)

lambda.by.effect <- matrix(c('Movie Effect',
                             'User Effect', 
                             'Genre Effect', 
                             'Year Effect', 
                             best.movie.tuner, 
                             best.user.tuner, 
                             best.genre.tuner, 
                             best.year.tuner), nrow = 4)

colnames(lambda.by.effect) <- c('Effect Type', 'Lambda')
lambda.by.effect <- knitr::kable(lambda.by.effect)


# rmses on the validation set

#extracting year for validation set

year.validation <- validation %>% mutate(year = str_extract(title, pattern =  "\\(\\d{4}\\)")) %>% 
  mutate(year = str_remove(year, '\\(')) %>% 
  mutate(year = str_remove(year, '\\)')) %>%
  mutate(year = as.numeric(year))%>%
  select(movieId,year) %>% distinct()


#calculating rmse for validation set - for final rmse, the model effects are calculated using the whole edx 
#data set in order to improve accuracy. 

validation.final.model.rmse <- validation %>%
  left_join(year.validation, by = 'movieId') %>%
  left_join(edx.movie.effect, by = 'movieId') %>%
  left_join(edx.user.effect, by = 'userId') %>%
  left_join(edx.genre.effect, by = 'genres') %>%
  left_join(edx.year.effect, by = 'year') %>%
  mutate(me = ifelse(is.na(me), 0, me)) %>%
  mutate(predicted = mu.edx + me + ue + ge + ye) %>%
  summarize(rmse.validation = sqrt(mean((rating - predicted)^2)))

validation.final.model.rmse
