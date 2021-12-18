head(cars)
# scatterplot
scatter.smooth(x=cars$speed, y=cars$dist, main="Dist ~ Speed")
#Build Linear Model:
# build linear regression model on full data
linearMod <- lm(dist ~ speed, data=cars)
print(linearMod)
Linear Regression Diagnostics
summary(linearMod)
