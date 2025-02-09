# TeslaStock
Work for my Founations of ML class

# Packages Installed

    "connectorx>=0.4.1",
    "matplotlib>=3.10.0",
    "nbformat>=5.10.4",
    "plotly>=6.0.0",
    "polars>=1.19.0",
    "pyarrow>=19.0.0",
    "sqlalchemy>=2.0.37",
 # Dependencies



    "ipykernel>=6.29.5",
    "jax[cuda12]>=0.4.38",
    "keras>=3.8.0",
    "scikit-learn>=1.6.1",
    "torch>=2.5.1",
    "torchvision>=0.20.1",


# Project 1: Predicting Tesla stock using

# Elon’s Tweets

## 1 Introduction and Problem Formulation

**1.1. Background** :
My fascination with using sentiment analysis of tweets to make informed stock
movements all started when I stumbled upon the repo “trump2cash”. Trump2cash
tracked names of companies from Trump’s tweets, performed sentiment analysis using
Google's Natural Language API, then chose a trading strategy. Similar to the correlation
between the president’s tweets and the stock market, there is reportedly a strong
connection between CEO’s tweets and stock prices. With this knowledge we will study
the connection between Elon Musk’s tweets and stock price. He is a prime candidate
because of his recent average of 101 tweets per day, as well as the recent volatility in
the stock. To make this project easier I will be using a dataset that was created by a
group of students at Buffalo University asking the same question. They used VADER to
perform sentiment analysis and created a data set with valuable information. I will then
use regression for this project because I thought that it would be very valuable if I would
be able to predict the exact closing price of the stock.

**1.2 Impact:**
The biggest impact that an accurate model would make would be that it could
possibly work with sentiment analysis of Elon’s current tweets. If it is able to accurately
predict stock swings. Then anyone who runs it will be able to profit off of it. So if it
works, it will benefit those in the class who want to trade.

**1.3 Inputs and Outputs:**
The inputs will be the average of the daily sentiment scores and the previous
day's closing price. The outputs would be the predicted closing price of the stock. The
model will be trained on this criteria from 2018-2020.
I would hypothesize that the inputs can be used to make accurate predictions
because there appears to be a connection between the input and the output. The
repository TweetSentiAnalysis that I got the data set from also graphed a plot that
showed the rise and fall in stock prices based solely on when there is a positive or a
negative tweet. Forgive the blurry image, but I believe that if we have a model that fits
the data better, we can get better accuracy than just detecting if the sentiment is
positive or negative.

![Screenshot From 2025-02-09 00-17-53](https://github.com/user-attachments/assets/d940c055-3ac0-4bc9-97db-44f97598e6f5)

### 2. Methods and Experiments

**2.1 Source of data** : https://github.com/Anand-GitH/TweetSentiAnalysis/
I exported the data into a csv, but used the database in my code. I will include
both when submitting the project.
I did have to do some data sanitization. This was done in the SQL query portion
of my code. There were a few problems that I ran into when using the un-sanitized data
in my code. A tweet can be negative, positive or neutral. In the data set a neutral
sentiment was given a value of 1. My code was interpreting this as a very positive
sentiment. After re-assigning every neutral case to zero we began to see better results.
Another sanitization method that I did was take the average of each of the tweet
sentiments for each day. This would allow for a cleaner dataset for predictions.

**2.2 Test Set:**
I then set aside 10% of my data as a test set and named it X_test and Y_test
respectively. I then used those values to compare the final performance of my model

**2.3 Error function form:**
Since I am solving a regression problem, I want to use a mean squared error for
my error function. I did this by setting the scoring to neg_mean_squared_error in the
sklearn.model_cross val score. I made sure to be conscious of the negative output of
the error.

**2.4 Error Estimation Method:**
I chose the k-fold cross validation to estimate the value of my error function on
new data (Enew).

**2.5 Baseline Model:**
I used scikit-learn’s DummyRegressor model to act as a sanity check. I used their
mean function, which always predicts the mean of a training set. With the
DummyRegressor I estimated the error Enew using the k-fold method mentioned above to
be 4829.48787. Below is a graph of what this model looks like when compared to the
actual stock price.

![image](https://github.com/user-attachments/assets/f3205968-3a0b-47c1-8723-4a75047bed2e)



**2.6 K-Nearest-Neighbors algorithm:**
I then used the k-Nearest-Neighbors algorithm to develop a model. I did this by
performing a hyperparameter sweep to determine the best value for k. I performed the
sweep over the values 1-30. I found that the best model was found at a k value of 4
which resulted in an estimated Enew of 270.548.

![image](https://github.com/user-attachments/assets/4c4e27ad-4705-4f49-9385-ba05f0f4f819)


After training this model, I found that the final best model had a performance
using MSE of 317.401 1 when tested on the test set. This is what the model looks like
when it is against the true stock price.

![image](https://github.com/user-attachments/assets/09545cf2-4741-4e36-9c2a-4c4e88893e3b)


Wanting better performance I next swept over the metric and weights
hyperparameters. Now I am sweeping over k values 1-30, metrics (euclidean
manhattan, minkowski), and weights (uniform, distance). After sweeping through all 174
candidates, the best hyperparameter is a k of 8, euclidean metrics, and distance
weights. This had an estimated Enew of 253.5184.
![image](https://github.com/user-attachments/assets/313a327e-d363-4c94-8db5-11f74642a1ae)

When the full model was trained, it had a MSE of 271.0015 on the test set. Here
is what it looked like when compared to the actual stock prices. This model tracks the
actual stock price nicely, and only gets thrown off by swift market corrections, followed
by a surge in the market.

![image](https://github.com/user-attachments/assets/7709610e-76ae-4732-acbf-af44e1e93b67)

**2.7 Decision Trees algorithm:**
Next I used decision trees to develop a model. I chose to use squared_error and
absolute_error as my criterion loss functions for determining how to split regions. I
swept over the depth for all criteria and both hyperparameter combinations using the
values (3, 5, 10, 15, None/unlimited depth).
For each loss function I then performed a hyperparameter sweep to determine
the best hyperparameter combination. I will sweep all over the min_samples_split for
values (3,5,10,20) and min_samples_leaf for values (1, 5, 10, 20). In total I will be
testing two criteria (squared error, and absolute error), 5 max tree depths, and 4 values
for min_samples_split and min_samples_leaf.
Using this I was able to generate a plot for each criteria showing the
hyperparameter sweeps by evaluating error using k-fold validation method.

![image](https://github.com/user-attachments/assets/a65f6fd1-3865-495c-9090-0aa4b94636be)

![image](https://github.com/user-attachments/assets/1676f8e7-76fc-40be-a656-fbb0905342b6)


Below is the best decision tree model trained and its predictions graphed on to
the true stock price. It uses a depth of 5, a min_samples_split of 3, a min_samples leaf
of 1, and a criterion of squared error. Using this I was able to get a cross validation
score of 197.07. You can see that it tends to lag below the peaks of the actual stock, but
tends to do a good job at predicting the market. After being trained it was able to get a
MSE of 237.72 on the test set. This was the best yet! The difference between the final
tested error and the cross validation score is possibly because of some slight overfitting.

![image](https://github.com/user-attachments/assets/79e7cea5-c9ef-4bbc-b2f2-db8fb6a826f3)


**2.8 Performance Evaluation:**
The final models tested on the test set for the baseline had a Mean Squared
Error of 6020, the K-Nearest-Neighbor had an error of 271, and the Decision Tree
Model had an error of 237. Both the KNN and Decision tree were able to beat the
Dummy model by miles, this means that they are a good representation of the
estimated stock price.

![image](https://github.com/user-attachments/assets/7d57f799-35d4-4036-89ce-2706a13b1384)


## 3. Discussion and Conclusion

**3.1 Model Performance Analysis:**
In the end, the Decision Tree performs best. Because we were measuring final
performance using mean squared error, the model emphasizes large errors by squaring
them. This means that the large error at the end of the KNN model is squared and even
though it outperformed the Decision Tree for most of the months, it still has a larger
error. If you used this KNN model you would have missed some of the massive growth
leading into 2020, and for this reason the Decision Tree would be the best performing
model.

![image](https://github.com/user-attachments/assets/f4411372-afb3-4f57-a927-59954bb23d74)


Another method that I thought that would be useful for measuring performance
would be to ignore the amount that each model was off, but map if it was able to predict
the direction of the stock “Going up”/ “Going Down” correctly. This gives us a categorical
way to analyse how the model was performing. If you analyse the two models below this
way you see that the Decision Tree was able to correctly predict the direction 61.09% of
the time, while the KNN was able to get it 65.91%. While I thought this would be a great
classification method, it can be slightly misleading. The model should not be judged on
how often it gets things right, but also by the magnitude that it wins or loses. Therefore,
while trend direction is interesting, the primary metric is Mean Squared Error.

![image](https://github.com/user-attachments/assets/e439d56c-d4b1-44d5-90d8-c1e6eafe664f)

![image](https://github.com/user-attachments/assets/2f969e8c-3582-4a56-88cb-9dfb5e250668)


**3.2 Model vs Baseline:**
Admittedly when I very first went to make the model, I was only using the Tweet
Scores to predict the market, and not including the previous day's closing price as an
input. This leads to the baseline model being close to the performance of the best
model (Which had an error of around 5,000). With that issue fixed, the best model did
significantly better than the baseline that I trained. Since the stock market is very fluid,
and the stock for Tesla rose considerably, simply taking the mean of the stock led to
large errors. In the graph below, you can see the Decision Tree, along with the other
models closely following the actual daily price.

![image](https://github.com/user-attachments/assets/3a10a3f7-cbb7-4b8b-8774-6f464243c857)


**3.3 Confidence in real-world Applications:**
I would like to preface my confidence in the model by first acknowledging its
shortcomings. First off, the data that I used has a few glaring flaws. Since it just
aggregates the tweet data by day, and only looks at day to day stock prices which could
be too broad. The models do not take into account any of the economy at large, any of
the market context, and do not output any confidence intervals. The decision tree
seems to tend to overfit to the training data, making it quite sensitive to small changes.
This could lead to very unstable trading patterns. The K-Nearest Neighbors tends to
lean more upon the previous day's stock price to predict the trend, this makes the result
fall short when the market doubles back on itself. Even though the models have
significant shortcomings, I am not confident but slightly satisfied with the result because
it would have still earned some money due to its over 50 percent accuracy on trends.
Using a trading strategy based off of this information would be slightly better than a coin
toss. The model would be able to earn money but would be risky for not much reward.
In the end, I would love to use this in the real world. I would need to test it on
more recent data, and would like to increase the granularity of the inputs. Once this is
done, I would feel comfortable giving the model only the tiniest amount of money to test
the concept based on its “slightly better than guessing” performance. Other future
concepts worth exploring would be to only make an estimate only if the tweet mentions
the company or provide it with more information to overcome the lack of economic
context.
