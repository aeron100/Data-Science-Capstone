#Data Science MovieLens Project Code#
#Project setup#

library(tidyverse)
library(dplyr)
library(DescTools)
library(caret)
library(data.table)

#I downloaded the MovieLens file which is a beast-sized file
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


# The validation set aka training will be set at 10% of the MovieLens dataset
set.seed(1, sample.kind="Rounding")

# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx dateset
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

#I added rows removed from validation set back into edx datset
removed <- anti_join(temp, validation)

edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#Descriptive statistics#
#I calculated the overall summary statistics for the edx dataset 
summary(edx)

# I calculated the mean rating on the training dataset
mu <- mean(edx$rating)
mu

# I calculated the standard deviation rating on the training dataset
stdev <- sd(edx$rating)
stdev

# I calculated the frequency and created a visual representation of the distribution of movie ratings 
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "blue") + 
  scale_x_log10() + 
  ggtitle("Movies")

#Data model framework#

# I used the mean of the training set to calculate Root Mean Square Error or RMSE to predict unknown ratings
RMSE(validation$rating, mu)

# I created and added the average ranking term, called meanrank
meanrank <- edx %>%
  group_by(movieId) %>%
  summarize(meanrank = mean(rating - mu))

# I used mu and meanrank to predict all unknown ratings 
predicted_ratings <- validation %>% 
  left_join(meanrank, by='movieId') %>%
  mutate(pred = mu + meanrank) %>%
  pull(pred) 

# I calculated the Root Mean Square Error of movie ranking effect
RMSE(validation$rating, predicted_ratings)

# I created a visual representation of the meanrank distribution
qplot(meanrank, data = meanrank, bins = 15, color = I("blue"))

# I created and added the term to represent user bias, called userbias
userbias <- edx %>% 
  left_join(meanrank, by='movieId') %>%
  group_by(userId) %>%
  summarize(userbias = mean(rating - mu - meanrank))

# I used the new variables to predict ratings which include meanrank and userbias
yhat_ratings <- validation %>% 
  left_join(meanrank, by='movieId') %>%
  left_join(userbias, by='userId') %>%
  mutate(pred = mu + meanrank + userbias) %>%
  pull(pred)

# I calculated the Root Mean Square Error of the movie ranking effect
RMSE(yhat_ratings, validation$rating)


#Regularize the model framework#

# I calculated the best lambda from a sequence
lambdas <- seq(from=0, to=10, by=0.25)

# I calculated the Root Mean Square Error of each lambda and repeated the earlier model development steps and included regularization for the various factors of the model
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  meanrank <- edx %>% 
    group_by(movieId) %>%
    summarize(meanrank = sum(rating - mu)/(n()+l))
  userbias <- edx %>% 
    left_join(meanrank, by="movieId") %>%
    group_by(userId) %>%
    summarize(userbias = sum(rating - meanrank - mu)/(n()+l))
  yhat_ratings <- validation %>% 
    left_join(meanrank, by = "movieId") %>%
    left_join(userbias, by = "userId") %>%
    mutate(pred = mu + meanrank + userbias) %>%
    pull(pred)
  return(RMSE(yhat_ratings, validation$rating))
})

# I created a visual representation using a quick plot of the Root Mean Square Error and lambdas
qplot(lambdas, rmses)

# print minimum RMSE 
min(rmses)

qplot(lambdas, rmses)
# print minimum RMSE 
min(rmses)


# Finalized model#
# The finalized the linear model by the minimizing lambda
lam <- lambdas[which.min(rmses)]

meanrank <- edx %>% 
  group_by(movieId) %>%
  summarize(meanrank = sum(rating - mu)/(n()+lam))
userbias <- edx %>% 
  left_join(meanrank, by="movieId") %>%
  group_by(userId) %>%
  summarize(userbias = sum(rating - meanrank - mu)/(n()+lam))

yhat_ratings <- validation %>% 
  left_join(meanrank, by = "movieId") %>%
  left_join(userbias, by = "userId") %>%
  mutate(pred = mu + meanrank + userbias) %>%
  pull(pred)

#The final model output Root Mean Square Error of the model predictions
RMSE(yhat_ratings, validation$rating)
