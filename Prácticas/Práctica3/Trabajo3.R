## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
set.seed(5552368)
library(ROCR)

## ----Carga.ISLR----------------------------------------------------------
library(ISLR)

## ------------------------------------------------------------------------
pairs(Auto)

## ------------------------------------------------------------------------
boxplot(mpg~cut(displacement, breaks = 10),data = Auto)

## ------------------------------------------------------------------------
boxplot(mpg~cut(horsepower, breaks = 10),data = Auto)

## ------------------------------------------------------------------------
boxplot(mpg~cut(weight, breaks = 10),data = Auto)

## ------------------------------------------------------------------------
idx.train = sample(nrow(Auto),size = nrow(Auto)*0.8)
Auto.train = Auto[idx.train,c("mpg","displacement","horsepower","weight")]
Auto.test = Auto[-idx.train,c("mpg","displacement","horsepower","weight")]

## ------------------------------------------------------------------------
Auto.train = data.frame(mpg01=sign(Auto.train$mpg>=median(Auto.train$mpg))*2-1, Auto.train)
Auto.test = data.frame(mpg01=sign(Auto.test$mpg>=median(Auto.train$mpg))*2-1, Auto.test)

## ------------------------------------------------------------------------
modelo.RegLog = glm(mpg01 ~ displacement + horsepower + weight, data = Auto.train, start=c(log(mean(Auto.train$mpg)),0,0,0))

## ------------------------------------------------------------------------
prediccion.glm = predict(modelo.RegLog,newdata = Auto.test)

## ------------------------------------------------------------------------
sum((sign(-prediccion.glm*Auto.test$mpg01)+1)/2)/nrow(Auto.test) * 100

## ------------------------------------------------------------------------
Auto.train.knn = scale(Auto.train[,c("displacement","horsepower","weight")])
media  = attr(Auto.train.knn, "scaled:center")
escala = attr(Auto.train.knn, "scaled:scale") 
Auto.test.knn= scale(Auto.test[,c("displacement","horsepower","weight")],media,escala)

Auto.train.knn = data.frame(mpg01=sign(Auto.train$mpg>=median(Auto.train$mpg))*2-1, Auto.train.knn)
Auto.test.knn = data.frame(mpg01=sign(Auto.test$mpg>=median(Auto.train$mpg))*2-1, Auto.test.knn)

## ------------------------------------------------------------------------
library(class)
prediccion.knn = knn(Auto.train.knn[,c("displacement","horsepower","weight")],Auto.test.knn[,c("displacement","horsepower","weight")], Auto.train.knn[,"mpg01"],k = 8)


## ------------------------------------------------------------------------
sum(prediccion.knn != Auto.test.knn$mpg01)/nrow(Auto.test.knn) * 100

## ------------------------------------------------------------------------
for(k in 1:20){
  prediccion.knn = knn(Auto.train.knn[,c("displacement","horsepower","weight")],Auto.test.knn[,c("displacement","horsepower","weight")], Auto.train.knn[,"mpg01"],k = k)
  error = sum(prediccion.knn != Auto.test.knn$mpg01)/nrow(Auto.test.knn) * 100
  cat(paste0("K: ",k, " \tError: ",error,"\n"))
}

## ----warning=FALSE-------------------------------------------------------
library(e1071)
tune.values = tune.knn(x = Auto.train.knn[,c("displacement","horsepower","weight")], y = as.factor(Auto.train.knn[,"mpg01"]),k = 1:20,tunecontrol = tune.control(sampling = "cross"), cross = 10)
summary(tune.values)

## ------------------------------------------------------------------------
prediccion.knn = knn(Auto.train.knn[,c("displacement","horsepower","weight")],Auto.test.knn[,c("displacement","horsepower","weight")], Auto.train.knn[,"mpg01"],k = 8, prob = TRUE)

prob <- attr(prediccion.knn, "prob")
prob <- 2*ifelse(prediccion.knn == "-1", 1-prob, prob) - 1

## ------------------------------------------------------------------------
pred <- prediction(prediccion.glm, Auto.test$mpg01)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))

## ------------------------------------------------------------------------
pred <- prediction(prob, Auto.test.knn$mpg01)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))

## ----warning=FALSE-------------------------------------------------------
library(caret)
folds = createFolds(1:nrow(Auto), k = 5)

error.log = vector("numeric",5)
error.knn = vector("numeric",5)

for(i in 1:5){
  Auto.train = Auto[-folds[[i]],c("mpg","displacement","horsepower","weight")]
  Auto.test = Auto[folds[[i]],c("mpg","displacement","horsepower","weight")]
  
  Auto.train = data.frame(mpg01=sign(Auto.train$mpg>=median(Auto.train$mpg))*2-1, Auto.train)
  Auto.test = data.frame(mpg01=sign(Auto.test$mpg>=median(Auto.train$mpg))*2-1, Auto.test)
  
  Auto.train.knn = scale(Auto.train[,c("displacement","horsepower","weight")])
  media  = attr(Auto.train.knn, "scaled:center")
  escala = attr(Auto.train.knn, "scaled:scale") 
  Auto.test.knn= scale(Auto.test[,c("displacement","horsepower","weight")],media,escala)
  
  Auto.train.knn = data.frame(mpg01=sign(Auto.train$mpg>=median(Auto.train$mpg))*2-1, Auto.train.knn)
  Auto.test.knn = data.frame(mpg01=sign(Auto.test$mpg>=median(Auto.train$mpg))*2-1, Auto.test.knn)
  
  #Regresion logistica
  modelo.RegLog = glm(mpg01 ~ displacement + horsepower + weight, data = Auto.train, start=c(log(mean(Auto.train$mpg)),0,0,0))
  prediccion.glm = predict(modelo.RegLog,newdata = Auto.test)
  error.log[i] = sum((sign(-prediccion.glm*Auto.test$mpg01)+1)/2)/nrow(Auto.test) * 100
  #Clasificacion K-NN
  prediccion.knn = knn(Auto.train.knn[,c("displacement","horsepower","weight")],Auto.test.knn[,c("displacement","horsepower","weight")], Auto.train.knn[,"mpg01"],k = 8)
  error.knn[i] = sum(prediccion.knn != Auto.test.knn$mpg01)/nrow(Auto.test.knn) * 100
}

cat(paste0("Error medio cometido con regresión logística: ",mean(error.log)))
cat(paste0("Error medio cometido con K-NN: ",mean(error.knn)))

## ------------------------------------------------------------------------
library(MASS)
idx.train = sample(nrow(Boston),size = nrow(Boston)*0.8)
Boston.train = Boston[idx.train,]
Boston.test = Boston[-idx.train,]

## ----warning=FALSE-------------------------------------------------------
library(glmnet)
prediccion.lasso <- glmnet(x = as.matrix(Boston.train[,-1]), y = Boston.train[,1])
cv = cv.glmnet(x = as.matrix(Boston.train[,-1]), y = Boston.train[,1], alpha = 1)

## ------------------------------------------------------------------------
lasso.coef = predict(prediccion.lasso,type="coefficients", s = cv$lambda.min)[1:14,]
coef = abs(lasso.coef)>0.5
coef
Boston.train[1,coef]

## ------------------------------------------------------------------------
wd = glmnet(x = as.matrix(Boston.train[,coef]), y = Boston.train[,1], alpha = 0)

## ------------------------------------------------------------------------
prediction = predict(wd, newx = as.matrix(Boston.test[,coef]),s = cv$lambda.min, type = "response")
error = mean((prediction - Boston.test[,1])^2)
cat(paste0("Error cometido: ",error))

## ------------------------------------------------------------------------
crim01 = sign(Boston[,1] >= median(Boston[,1]))*2-1
new.crim = crim01[idx.train]
Boston.train.01 = data.frame(Boston.train, new.crim)
new.crim = crim01[-idx.train]
Boston.test.01  = data.frame(Boston.test, new.crim)

## ------------------------------------------------------------------------
model.svm = svm(new.crim ~ chas+nox+rm+dis+rad, data = Boston.train.01, kernel = "linear")
prediction = predict(model.svm, newdata = Boston.test.01)

(sum((sign(-prediction*Boston.test.01$new.crim)+1)/2)/nrow(Boston.test.01)) *100

## ------------------------------------------------------------------------
model.svm = svm(new.crim ~ chas+nox+rm+dis+rad, data = Boston.train.01,kernel = "polynomial")
prediction = predict(model.svm, newdata = Boston.test.01)

(sum((sign(-prediction*Boston.test.01$new.crim)+1)/2)/nrow(Boston.test.01)) *100

## ------------------------------------------------------------------------
model.svm = svm(new.crim ~ chas+nox+rm+dis+rad, data = Boston.train.01,kernel = "radial")
prediction = predict(model.svm, newdata = Boston.test.01)

(sum((sign(-prediction*Boston.test.01$new.crim)+1)/2)/nrow(Boston.test.01)) *100

## ----warning=FALSE-------------------------------------------------------
folds = createFolds(1:nrow(Boston), k = 5)

error.in = vector("numeric",5)
error.test = vector("numeric",5)

for(i in 1:5){
  #Generar particion
  Boston.train = Boston[-folds[[i]],]
  Boston.test = Boston[folds[[i]],]
  
  crim01 = sign(Boston[,1] >= median(Boston[,1]))*2-1
  new.crim = crim01[-folds[[i]]]
  Boston.train.01 = data.frame(Boston.train, new.crim)
  new.crim = crim01[folds[[i]]]
  Boston.test.01  = data.frame(Boston.test, new.crim)
  
  #Calcular el modelo
  model.svm = svm(new.crim ~ chas+nox+rm+dis+rad, data = Boston.train.01,kernel = "radial")
  
  #Error de test
  prediction = predict(model.svm, newdata = Boston.test.01)
  error.test[1] = (sum((sign(-prediction*Boston.test.01$new.crim)+1)/2)/nrow(Boston.test.01)) *100
  
  #Error de train
  prediction = predict(model.svm, newdata = Boston.train.01)
  error.test[1] = (sum((sign(-prediction*Boston.train.01$new.crim)+1)/2)/nrow(Boston.train.01)) *100
}

cat(paste0("Error medio en train: ",mean(error.in)))
cat(paste0("Error medio en test: ",mean(error.test)))

## ------------------------------------------------------------------------
library(MASS)
idx.train = sample(nrow(Boston),size = nrow(Boston)*0.8)
Boston.train = Boston[idx.train,]
Boston.test = Boston[-idx.train,]

## ------------------------------------------------------------------------
library(randomForest)
random.forest = randomForest(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data = Boston.train, mtry = 13)
pred = predict(random.forest,Boston.test)
error = sum(abs(Boston.test["medv"]-pred))/nrow(Boston.test)
cat(paste0("El error obtenido es: ",error))

## ------------------------------------------------------------------------
random.forest = randomForest(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data = Boston.train)
pred = predict(random.forest,Boston.test)
error = sum(abs(Boston.test["medv"]-pred))/nrow(Boston.test)
cat(paste0("El error obtenido es: ",error))

## ------------------------------------------------------------------------
library(gbm)
boost = gbm(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data = Boston.train, distribution = "gaussian")
pred = predict(boost,Boston.test,n.trees = 100)
error = sum(abs(Boston.test["medv"]-pred))/nrow(Boston.test)
cat(paste0("El error obtenido es: ",error))

## ------------------------------------------------------------------------
idx.train = sample(nrow(OJ),size = 800)
OJ.train = OJ[idx.train,]
OJ.test = OJ[-idx.train,]

## ----warning=FALSE-------------------------------------------------------
library(tree)
arbol = tree(Purchase~WeekofPurchase+StoreID+PriceCH+PriceMM+DiscCH+DiscMM+SpecialCH+SpecialMM+LoyalCH+SalePriceMM+SalePriceCH+PriceDiff+Store7+PctDiscMM+PctDiscCH+ListPriceDiff+STORE, data = OJ.train)

## ------------------------------------------------------------------------
summary(arbol)

## ------------------------------------------------------------------------
plot(arbol, uniform=TRUE, 
  	main="Árbol de clasificación")
text(arbol, use.n=TRUE, all=TRUE, cex=.8)

## ------------------------------------------------------------------------
probabilidad.prediccion = predict(arbol,OJ.test)
OJ.prediction= vector(length = nrow(probabilidad.prediccion))
OJ.prediction[probabilidad.prediccion[,1]>0.5] = "CH"
OJ.prediction[probabilidad.prediccion[,1]<=0.5] = "MM"
confusionMatrix(OJ.prediction,OJ.test$Purchase)

## ------------------------------------------------------------------------
tr0.cv = cv.tree(arbol)
plot.tree.sequence(tr0.cv)

