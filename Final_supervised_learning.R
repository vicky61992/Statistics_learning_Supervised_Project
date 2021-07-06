# Load required libraries

library("leaps")
library("ggplot2")
library("reshape2")
library("MASS")
library("ggcorrplot")
library("plotmo")
library("dplyr")
library("gridExtra")
library("Simpsons")
library("GGally")
library("memisc")
library("pander")
library("caret")
library("glmnet")
library("mlbench")
library("psych")
library(data.table) # used for reading and manipulation of data
library(caret)      # used for modeling
library(xgboost)    # used for building XGBoost model
library(e1071)      # used for skewness
library(cowplot)    # used for combining multiple plots 
library(corrplot)





# Data importing and processing

getwd()
setwd("C:/Users/VSBAG/Desktop/DSE_Milan/3rd_sem_subject/ML&SL/SL_Project/My_Supervised_LEarning/S_SL_Project")
# Data Loading
train <- fread("https://raw.githubusercontent.com/vicky61992/Statistics_learning_Supervised_Project/main/Train.csv")
test <- fread("https://raw.githubusercontent.com/vicky61992/Statistics_learning_Supervised_Project/main/Test.csv")
submission <- fread("https://raw.githubusercontent.com/vicky61992/Statistics_learning_Supervised_Project/main/sample_submission.csv")
#Dimensions of Data
dim(train)
dim(test)

#Structure of Data
str(train)
str(test)

#Features of Data
head(train)
head(test)
names(train)
names(test)


# summary 
summary(train)
summary(test)

#Null value

sapply(train,function(x)sum(is.na(x)))
sapply(test,function(x)sum(is.na(x)))

#Combine Train and Test (To save TIME)

test[,Item_Outlet_Sales := NA]
combi = rbind(train, test) # combining train and test datasets
dim(combi)


#Exploratory Data Analysis(EDA)

ggplot(train) + geom_histogram(aes(train$Item_Outlet_Sales), binwidth = 100, fill = "pink") +
  xlab("Item_Outlet_Sales")


#Independent Variables(numeric variables) on Train data

p1 = ggplot(combi) + geom_histogram(aes(Item_Weight), binwidth = 0.5, fill = "blue")
p2 = ggplot(combi) + geom_histogram(aes(Item_Visibility), binwidth = 0.005, fill = "blue")
p3 = ggplot(combi) + geom_histogram(aes(Item_MRP), binwidth = 1, fill = "blue")
plot_grid(p1, p2, p3, nrow = 1) # plot_grid() from cowplot package


#Notes
#1There seems to be no clear-cut pattern in Item_Weight.
#2Item_Visibility is right-skewed and should be transformed to curb its skewness.
#3We can clearly see 4 different distributions for Item_MRP. It is an interesting insight.

#Independent Variables(categorical variables) 

ggplot(combi%>% group_by(Item_Fat_Content) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral1")


#Cobination of Low fat & Reg

combi$Item_Fat_Content[combi$Item_Fat_Content == "LF"] = "Low Fat"
combi$Item_Fat_Content[combi$Item_Fat_Content == "low fat"] = "Low Fat"
combi$Item_Fat_Content[combi$Item_Fat_Content == "reg"] = "Regular"

#Recrate Independent Variables(categorical variables)
ggplot(combi %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral1")

## check the other categorical variables

# plot for Item_Type
p4 = ggplot(combi %>% group_by(Item_Type) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Type, Count), stat = "identity", fill = "coral1") +
  xlab("") +
  geom_label(aes(Item_Type, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle("Item_Type")

# plot for Outlet_Identifier
p5 = ggplot(combi %>% group_by(Outlet_Identifier) %>% summarise(Count = n())) + 
  geom_bar(aes(Outlet_Identifier, Count), stat = "identity", fill = "coral1") +
  geom_label(aes(Outlet_Identifier, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# plot for Outlet_Size
p6 = ggplot(combi %>% group_by(Outlet_Size) %>% summarise(Count = n())) + 
  geom_bar(aes(Outlet_Size, Count), stat = "identity", fill = "coral1") +
  geom_label(aes(Outlet_Size, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

second_row = plot_grid(p5, p6, nrow = 1)
plot_grid(p4, second_row, ncol = 1)

#remaining categorical variables
# plot for Outlet_Establishment_Year
p7 = ggplot(combi %>% group_by(Outlet_Establishment_Year) %>% summarise(Count = n())) + 
  geom_bar(aes(factor(Outlet_Establishment_Year), Count), stat = "identity", fill = "coral1") +
  geom_label(aes(factor(Outlet_Establishment_Year), Count, label = Count), vjust = 0.5) +
  xlab("Outlet_Establishment_Year") +
  theme(axis.text.x = element_text(size = 8.5))
# plot for Outlet_Type
p8 = ggplot(combi %>% group_by(Outlet_Type) %>% summarise(Count = n())) + 
  geom_bar(aes(Outlet_Type, Count), stat = "identity", fill = "coral1") +
  geom_label(aes(factor(Outlet_Type), Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(size = 8.5))

# ploting both plots together
plot_grid(p7, p8, ncol = 2)

#Bivariate Analysis
#we'll explore the independent variables with respect to the target variable

train = combi[1:nrow(train)] # extracting train data from the combined data

#Target Variable vs Independent Numerical Variables
# Item_Weight vs Item_Outlet_Sales
p9 = ggplot(train) + 
  geom_point(aes(Item_Weight, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +
  theme(axis.title = element_text(size = 8.5))

# Item_Visibility vs Item_Outlet_Sales
p10 = ggplot(train) + 
  geom_point(aes(Item_Visibility, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +
  theme(axis.title = element_text(size = 8.5))

# Item_MRP vs Item_Outlet_Sales
p11 = ggplot(train) + 
  geom_point(aes(Item_MRP, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +
  theme(axis.title = element_text(size = 8.5))
second_row_2 = plot_grid(p10, p11, ncol = 2)
plot_grid(p9, second_row_2, nrow = 2)

#Target Variable vs Independent Categorical Variables

# Item_Type vs Item_Outlet_Sales
p12 = ggplot(train) + 
  geom_violin(aes(Item_Type, Item_Outlet_Sales), fill = "magenta") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 6),
        axis.title = element_text(size = 8.5))

# Item_Fat_Content vs Item_Outlet_Sales
p13 = ggplot(train) + 
  geom_violin(aes(Item_Fat_Content, Item_Outlet_Sales), fill = "magenta") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 8.5))

# Outlet_Identifier vs Item_Outlet_Sales
p14 = ggplot(train) + 
  geom_violin(aes(Outlet_Identifier, Item_Outlet_Sales), fill = "magenta") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 8.5))
second_row_3 = plot_grid(p13, p14, ncol = 2)
plot_grid(p12, second_row_3, ncol = 1)

#remaining variables
ggplot(train)+ geom_violin(aes(Outlet_Size, Item_Outlet_Sales),fill = "magenta")
p15 = ggplot(train) + geom_violin(aes(Outlet_Location_Type, Item_Outlet_Sales), fill = "magenta")
p16 = ggplot(train) + geom_violin(aes(Outlet_Type, Item_Outlet_Sales), fill = "magenta")
plot_grid(p15, p16, ncol = 1)


#Missing Value Treatment

#There are different methods to treat missing values based on the problem and the data. Some of the common techniques are as follows:

#1.Deletion of rows: In train dataset, observations having missing values in any variable are deleted. The downside of this method is the loss of information and drop in prediction power of model.

#2.Mean/Median/Mode Imputation: In case of continuous variable, missing values can be replaced with mean or median of all known values of that variable. For categorical variables, we can use mode of the given values to replace the missing values.

#3.Building Prediction Model: We can even make a predictive model to impute missing data in a variable. Here we will treat the variable having missing data as the target variable and the other variables as predictors. We will divide our data into 2 datasets-one without any missing value for that variable and the other with missing values for that variable. The former set would be used as training set to build the predictive model and it would then be applied to the latter set to predict the missing values.

#find missing values in a variable.

sum(is.na(combi$Item_Weight))

#Imputing Missing Value

missing_index = which(is.na(combi$Item_Weight))
for(i in missing_index){
  
  item = combi$Item_Identifier[i]
  combi$Item_Weight[i] = mean(combi$Item_Weight[combi$Item_Identifier == item], na.rm = T)
}

#Cross Check

sum(is.na(combi$Item_Weight))
sapply(combi,function(x)sum(is.na(x)))

#Replacing 0's in Item_Visibility variable

ggplot(combi) + geom_histogram(aes(Item_Visibility), bins = 100)

#replace the zeroes

zero_index = which(combi$Item_Visibility == 0)
for(i in zero_index){
  
  item = combi$Item_Identifier[i]
  combi$Item_Visibility[i] = mean(combi$Item_Visibility[combi$Item_Identifier == item], na.rm = T)
  
}

ggplot(combi) + geom_histogram(aes(Item_Visibility), bins = 100)



#Feature Engineering

#will create the following new features:

#1Item_Type_new: Broader categories for the variable Item_Type.
#2Item_category: Categorical variable derived from Item_Identifier.
#3Outlet_Years: Years of operation for outlets.
#4price_per_unit_wt: Item_MRP/Item_Weight
#5Item_MRP_clusters: Binned feature for Item_MRP.

#Item_Type variable and classify the categories into perishable and non_perishable as per our understanding and make it into a new feature.

perishable = c("Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood")
non_perishable = c("Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene", "Household", "Soft Drinks")

# create a new feature 'Item_Type_new'
combi[,Item_Type_new := ifelse(Item_Type %in% perishable, "perishable", ifelse(Item_Type %in% non_perishable, "non_perishable", "not_sure"))]

#Let's compare Item_Type with the first 2 characters of Item_Identifier, i.e., 'DR', 'FD', and 'NC'. These identifiers most probably stand for drinks, food, and non-consumable.

table(combi$Item_Type, substr(combi$Item_Identifier, 1, 2))

# Item_category Created

combi[,Item_category := substr(combi$Item_Identifier, 1, 2)]

#Outlet_Years (years of operation) and price_per_unit_wt (price per unit weight).

combi$Item_Fat_Content[combi$Item_category == "NC"] = "Non-Edible"

# years of operation for outlets
combi[,Outlet_Years := 2013 - Outlet_Establishment_Year]
combi$Outlet_Establishment_Year = as.factor(combi$Outlet_Establishment_Year)

# Price per unit weight
combi[,price_per_unit_wt := Item_MRP/Item_Weight]

#the Item_MRP vs Item_Outlet_Sales plot, we saw Item_MRP was spread across in 4 chunks. Now let's assign a label to each of these chunks and use this label as a new variable.

# creating new independent variable - Item_MRP_clusters
combi[,Item_MRP_clusters := ifelse(Item_MRP < 69, "1st", 
                                   ifelse(Item_MRP >= 69 & Item_MRP < 136, "2nd",
                                          ifelse(Item_MRP >= 136 & Item_MRP < 203, "3rd", "4th")))]

#Encoding Categorical Variables

#Most of the machine learning algorithms produce better result with numerical variables only. So, it is essential to treat the categorical variables present in the data

#We will use 2 techniques - Label Encoding and One Hot Encoding.

#1Label encoding simply means converting each category in a variable to a number. It is more suitable for ordinal variables - categorical variables with some order.

#2In One hot encoding, each category of a categorical variable is converted into a new binary column (1/0).

#Label encoding for the categorical variables
str(combi)

combi[,Outlet_Size_num := ifelse(Outlet_Size == "Small", 0,
                                 ifelse(Outlet_Size == "Medium", 1,2))]
combi[,Outlet_Location_Type_num := ifelse(Outlet_Location_Type == "Tier 3", 0,
                                          ifelse(Outlet_Location_Type == "Tier 2", 1, 2))]
# removing categorical variables after label encoding
combi[, c("Outlet_Size", "Outlet_Location_Type") := NULL]

#One hot encoding for the categorical variable

ohe = dummyVars("~.", data = combi[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")], fullRank = T)
ohe_df = data.table(predict(ohe, combi[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")]))
combi = cbind(combi[,"Item_Identifier"], ohe_df)



#PreProcessing Data


skewness(combi$Item_Visibility); skewness(combi$price_per_unit_wt)


#Removing Skewness

combi[,Item_Visibility := log(Item_Visibility + 1)] # log + 1 to avoid division by zero
combi[,price_per_unit_wt := log(price_per_unit_wt + 1)]



#Scaling numeric predictors

#Let's scale and center the numeric variables to make them have a mean of zero, standard deviation of one and scale of 0 to 1. Scaling and centering is required for linear regression models.

num_vars = which(sapply(combi, is.numeric)) # index of numeric features
num_vars_names = names(num_vars)
combi_numeric = combi[,setdiff(num_vars_names, "Item_Outlet_Sales"), with = F]
prep_num = preProcess(combi_numeric, method=c("center", "scale"))
combi_numeric_norm = predict(prep_num, combi_numeric)

combi[,setdiff(num_vars_names, "Item_Outlet_Sales") := NULL] # removing numeric independent variables
combi = cbind(combi, combi_numeric_norm)



#Splitting the combined data combi back to train and test set.

train = combi[1:nrow(train)]
test = combi[(nrow(train) + 1):nrow(combi)]
test[,Item_Outlet_Sales := NULL] # removing Item_Outlet_Sales as it contains only NA

# Correlation Plot

cor_train = cor(train[,-c("Item_Identifier")])
corrplot(cor_train, method = "pie", type = "lower", tl.cex = 0.9)

sapply(train,function(x)sum(is.na(x)))


#Model Building

#Linear Regression

#Building Model

linear_reg_mod1 = lm(Item_Outlet_Sales ~ ., data = train[,-c("Item_Identifier")])

summary(linear_reg_mod1)

predtest<- predict(linear_reg_mod1,test[,-c("Item_Identifier")])


#Making Predictions on test Data

head(predtest)
predtest1<-data.frame(predtest)
final_data<- cbind(test,predtest1)
write.csv(final_data,"predicted.csv")


# preparing dataframe for submission and writing it in a csv file

set.seed(1234)
my_control = trainControl(method="cv", number=5)
linear_reg_mod = train(x = train[,-c("Item_Identifier", "Item_Outlet_Sales")], y = train$Item_Outlet_Sales,
                       method='glmnet', trControl= my_control)
print("5- fold cross validation scores:")
predtest1<- predict(linear_reg_mod,test[,-c("Item_Identifier")])

eval_results <- function(true, predicted, df) {
  
  SSE <- sum((predicted - true)^2)
  
  SST <- sum((true - mean(true))^2)
  
  R_square <- 1 - SSE / SST
  
  RMSE = sqrt(SSE/nrow(df))
  # Model performance metrics
  
  data.frame(
    
    RMSE = RMSE,
    
    Rsquare = R_square
    
  )
  
  
  
}

# Prediction and evaluation on train data

predictions_train <- predict(linear_reg_mod, s = optimal_lambda, newx = x)

eval_results(train$Item_Outlet_Sales, predictions_train, train)

summary(linear_reg_mod)
print(round(linear_reg_mod$resample$RMSE, 2))
submission$Item_Outlet_Sales = predict(linear_reg_mod, test[,-c("Item_Identifier")])
write.csv(submission, "Linear_Reg_submit_2_21_Apr_18.csv", row.names = F)

# mean validation score
mean(linear_reg_mod$resample$RMSE)


#Lasso Regression

set.seed(1235)
my_control = trainControl(method="cv", number=5)
Grid = expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0002))

lasso_linear_reg_mod = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")], y = train$Item_Outlet_Sales,
                             method='glmnet', trControl= my_control, tuneGrid = Grid)

predtest2<- predict(lasso_linear_reg_mod,test[,-c("Item_Identifier")])

# Prediction and evaluation on train data

predictions_train <- predict(lasso_linear_reg_mod, s = optimal_lambda, newx = x)

eval_results(train$Item_Outlet_Sales, predictions_train, train)


#Ridge Regression

set.seed(1236)
my_control = trainControl(method="cv", number=5)
Grid = expand.grid(alpha = 0, lambda = seq(0.001,0.1,by = 0.0002))

ridge_linear_reg_mod = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")], y = train$Item_Outlet_Sales,
                             method='glmnet', trControl= my_control, tuneGrid = Grid)

predtest3<- predict(ridge_linear_reg_mod,test[,-c("Item_Identifier")])


# Prediction and evaluation on train data

predictions_train <- predict(ridge_linear_reg_mod, s = optimal_lambda, newx = x)

eval_results(train$Item_Outlet_Sales, predictions_train, train)



#RandomForest 

set.seed(1237)
my_control = trainControl(method="cv", number=5)

tgrid = expand.grid(
  .mtry = c(3:10),
  .splitrule = "variance",
  .min.node.size = c(10,15,20)
)

rf_mod = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")], 
               y = train$Item_Outlet_Sales,
               method='ranger', 
               trControl= my_control, 
               tuneGrid = tgrid,
               num.trees = 400,
               importance = "permutation")

predtest4<- predict(rf_mod,test[,-c("Item_Identifier")])

# Prediction and evaluation on train data

predictions_train <- predict(rf_mod, s = optimal_lambda, newx = x)

eval_results(train$Item_Outlet_Sales, predictions_train, train)



# best model parameter
plot(rf_mod)


# Important variable 

plot(varImp(rf_mod))



#XGBoost

param_list = list(
  
  objective = "reg:linear",
  eta=0.01,
  gamma = 1,
  max_depth=6,
  subsample=0.8,
  colsample_bytree=0.5
)
dtrain = xgb.DMatrix(data = as.matrix(train[,-c("Item_Identifier", "Item_Outlet_Sales")]), label= train$Item_Outlet_Sales)
dtest = xgb.DMatrix(data = as.matrix(test[,-c("Item_Identifier")]))

#Cross Validation

set.seed(112)
xgbcv = xgb.cv(params = param_list, 
               data = dtrain, 
               nrounds = 1000, 
               nfold = 5, 
               print_every_n = 10, 
               early_stopping_rounds = 30, 
               maximize = F)

#Model Training

xgb_model = xgb.train(data = dtrain, params = param_list, nrounds = 429)


#Variable Importance

var_imp = xgb.importance(feature_names = setdiff(names(train), c("Item_Identifier", "Item_Outlet_Sales")), 
                         model = xgb_model)
xgb.plot.importance(var_imp)









































