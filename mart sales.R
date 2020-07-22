library(data.table)
library(ggplot2)
library(cowplot)
library(caret)
library(dplyr) 
library(xgboost)

train_data <- fread('C:/Users/Hshan/Downloads/train_v9rqX0R.csv')
test_data <- fread('C:/Users/Hshan/Downloads/test_AbJTz2l.csv')
test_data$Item_Outlet_Sales <- rep(0,nrow(test_data))

names(train_data)
cat_col <- c("Item_Fat_Content","Item_Type","Outlet_Size","Outlet_Location_Type", "Outlet_Type")
num_col <- c("Item_Weight","Item_Visibility","Item_MRP","Outlet_Establishment_Year")
identity_col <- c("Item_Identifier","Outlet_Identifier")
target_col <- "Item_Outlet_Sales"

#combine train data and test data:
data <- rbind(train_data, test_data)

tr_rows <- nrow(train_data)
tr_ndim <- dim(train_data); 

# only feature-weight has missing value, 17.2% 
# weight is continuous variable, checking on the distribution and decide on the imputation method

######################### Plotting analysis #########################
cont_p1 <- ggplot(train_data) + 
  geom_histogram(aes(Item_Weight), binwidth=0.5, fill = "blue") 
cont_p2 <- ggplot(train_data) + 
  geom_boxplot(aes(Item_Visibility), fill = "blue") ; cont_p2
cont_p3 <- ggplot(train_data) + 
  geom_boxplot(aes(Item_MRP), fill = "blue") 
plot_grid(cont_p1, cont_p2, cont_p3, nrow = 1) 

#boxplot for item_weight showed presence of outliers

train_data$Item_Fat_Content[train_data$Item_Fat_Content=='LF' | 
                              train_data$Item_Fat_Content=='low fat'] <- 'Low Fat'
train_data$Item_Fat_Content[train_data$Item_Fat_Content=='reg' | 
                              train_data$Item_Fat_Content=='regular'] <- 'Regular'

cat_p1 <- ggplot(train_data %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral1")

# plot for Outlet_Establishment_Year 
cat_p2 = ggplot(train_data %>% group_by(Outlet_Establishment_Year) %>% summarise(Count = n())) + 
  geom_bar(aes(factor(Outlet_Establishment_Year), Count), stat = "identity", fill = "coral1") + 
  geom_label(aes(factor(Outlet_Establishment_Year), Count, label = Count), vjust = 0.5) + 
  xlab("Outlet_Establishment_Year") +  
  theme(axis.text.x = element_text(size = 8.5)); cat_p2
# not the high value in 1985 and low value in 1998

# plot for Outlet_Type 
cat_p3 = ggplot(train_data %>% group_by(Outlet_Type) %>% summarise(Count = n())) +  
  geom_bar(aes(Outlet_Type, Count), stat = "identity", fill = "coral1") + 
  geom_label(aes(factor(Outlet_Type), Count, label = Count), vjust = 0.5) + 
  theme(axis.text.x = element_text(size = 8.5)); cat_p3
# note the highest value of count for supermarket type 1

plot_grid(cat_p2, cat_p3, ncol = 2)

# Item_Weight vs Item_Outlet_Sales 
weight_sales <- ggplot(train_data) + 
  geom_point(aes(Item_Weight, Item_Outlet_Sales), colour = "violet", alpha = 0.3) + 
  theme(axis.title = element_text(size = 8.5))
# Item_Visibility vs Item_Outlet_Sales 
visibility_sales <- ggplot(train_data) + 
  geom_point(aes(Item_Visibility, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +  
  theme(axis.title = element_text(size = 8.5))
# Item_MRP vs Item_Outlet_Sales 
mrp_sales <- ggplot(train_data) +
  geom_point(aes(Item_MRP, Item_Outlet_Sales), colour = "violet", alpha = 0.3) + 
  theme(axis.title = element_text(size = 8.5))
second_row_2 <- plot_grid(weight_sales, visibility_sales, ncol = 2) 
plot_grid(mrp_sales, second_row_2, nrow = 2)

# Item_Type vs Item_Outlet_Sales 
iType_sales <- ggplot(train_data) + 
  geom_violin(aes(Item_Type, Item_Outlet_Sales), fill = "magenta") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text = element_text(size = 6), axis.title = element_text(size = 8.5))
# Item_Fat_Content vs Item_Outlet_Sales 
iFat_sales <- ggplot(train_data) + 
  geom_violin(aes(Item_Fat_Content, Item_Outlet_Sales), fill = "magenta") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1),  axis.text = element_text(size = 8), axis.title = element_text(size = 8.5))
# Outlet_Identifier vs Item_Outlet_Sales 
iIdentifier_sales <- ggplot(train_data) + 
  geom_violin(aes(Outlet_Identifier, Item_Outlet_Sales), fill = "magenta") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text = element_text(size = 8), axis.title = element_text(size = 8.5))
second_row_3 <- plot_grid(p13, p14, ncol = 2) 
plot_grid(p12, second_row_3, ncol = 1)

oSize_sales <- ggplot(train_data) + 
  geom_violin(aes(Outlet_Size, Item_Outlet_Sales), fill = "magenta")

oLocation_sales <- ggplot(train) + 
  geom_violin(aes(Outlet_Location_Type, Item_Outlet_Sales), fill = "magenta") 
oType_sales <- ggplot(train) + 
  geom_violin(aes(Outlet_Type, Item_Outlet_Sales), fill = "magenta") 
plot_grid(oLocation_sales, oType_sales, ncol = 1)
###########################################################################

#define function: imputation for missing data in Item_weight
Weight_fillna <- function(train_data){
  missing_ind = which(is.na(train_data$Item_Weight)) 
  for(i in missing_ind){  
    item = train_data$Item_Identifier[i]  
    train_data$Item_Weight[i] = mean(train_data$Item_Weight[train_data$Item_Identifier == item], na.rm = T)
  }
  return (data.frame(train_data))
}
# na.rm=T means removing na when computing arithmetic function
data <- Weight_fillna(data)
data <- na.omit(data)

#note the visibility of 0
visibility_fillzero <- function(train_data){
  zero_ind = which(train_data$Item_Visibility==0)
  for(i in zero_ind){  
    item = train_data$Item_Identifier[i]  
    train_data$Item_Visibility[i] = mean(train_data$Item_Visibility[train_data$Item_Identifier == item], na.rm = T)
  }
  return (data.frame(train_data))
}

data <- visibility_fillzero(data)
outlet <- data$Outlet_Identifier
categorical_col = c(cat_col, 'Outlet_Identifier')
data$Item_Visibility <- log(1+ data$Item_Visibility)
data <- label_encoder(data, categorical_col)

#label encoding : 
label_encoder <- function(data, categorical_col){
  for (col in categorical_col){
    data[,col] <- as.integer(as.factor(data[,col]))
  }
  return (data.frame(data))
}


#splitting data:
train_data_1 <- data[1:tr_rows,]
test_data_1 <- data[(tr_rows+1):dim(data)[1],]
outlet <- outlet[(tr_rows+1):dim(data)[1]]

y_train <- train_data_1$Item_Outlet_Sales
x_train <- train_data_1 %>% select(-'Item_Identifier',-'Item_Outlet_Sales')


x_test <- test_data_1 %>% select(-'Item_Outlet_Sales',-'Item_Identifier')

#checking on the correlation between features or using lasso or ridge to penalize the correlation

param= list(objective = "reg:gamma",
                  eta=0.05, gamma = 3, max_depth=6, subsample=0.8,colsample_bytree=0.8,eval_metric = 'rmse')
dtrain = xgb.DMatrix(data = as.matrix(x_train), 
                     label= y_train) 
dtest = xgb.DMatrix(data = as.matrix(x_test))

set.seed(123) 
xgbcv = xgb.cv(params = param, data = dtrain,nrounds = 1000, nfold = 5, print_every_n = 20, 
               early_stopping_rounds = 30, maximize = F)
xgb_model = xgb.train(data = dtrain, params = param, nrounds = 1000)

y_test <- predict(xgb_model, dtest)
submission <- data.table('Item_Identifier' = test_data_1$Item_Identifier,
                         'Outlet_Identifier' = outlet,
                         'Item_Outlet_Sales' = y_test)


# TEST DATA
# FILL NA, REMOVE NA, FILL ZERO VISBILITY 
