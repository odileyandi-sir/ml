x=c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)
y=c(3,4,5,4,8,10,10,11,14,20,23,24,32,34,35,37,42,48,53,60)
train=data.frame(x,y)
plot(train,pch=16)
model <- lm(y ~ x, train)
abline(model)
library(e1071)
model_svm <- svm(y ~ x , train)
pred <- predict(model_svm, train)
points(train$x, pred, col = "blue", pch=4)
error <- model$residuals
lm_error <- sqrt(mean(error^2)) # 3.832974
error_2 <- train$y - pred
svm_error <- sqrt(mean(error_2^2)) # 2.696281
svm_tune <- tune(svm, y ~ x, data = train,
ranges = list(epsilon = seq(0,1,0.01), cost = 2^(2:9))

print(svm_tune)
best_mod <- svm_tune$best.model
best_mod_pred <- predict(best_mod, train)
error_best_mod <- train$y - best_mod_pred
best_mod_RMSE <- sqrt(mean(error_best_mod^2)) # 1.290738
plot(svm_tune)
plot(train,pch=16)
points(train$x, best_mod_pred, col = "blue", pch=4)
