# using dataset from the UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/)
titanicDF <- read.csv('http://math.ucdenver.edu/RTutorial/titanic.txt',sep='\t')

# creating new title feature
titanicDF$Title <- ifelse(grepl('Mr ',titanicDF$Name),'Mr',ifelse(grepl('Mrs ',titanicDF$Name),'Mrs',ifelse(grepl('Miss',titanicDF$Name),'Miss','Nothing')))
titanicDF$Title <- as.factor(titanicDF$Title)

# impute age to remove NAs
titanicDF$Age[is.na(titanicDF$Age)] <- median(titanicDF$Age, na.rm=T)

# reorder data set so target is last column
titanicDF <- titanicDF[c('PClass', 'Age',    'Sex',   'Title', 'Survived')]

# binarize all factors
library(caret)
titanicDummy <- dummyVars("~.",data=titanicDF, fullRank=F)
titanicDF <- as.data.frame(predict(titanicDummy,titanicDF))

# Let's take a peek at our data:
str(titanicDF)

# split data set into train and test portion
set.seed(1234)
splitIndex <- sample(nrow(titanicDF), floor(0.5*nrow(titanicDF)))
trainDF <- titanicDF[ splitIndex,]
testDF  <- titanicDF[-splitIndex,]

outcomeName <- 'Survived'
predictorNames <- setdiff(names(trainDF),outcomeName)

# transform outcome variable to text as this is required in caret for classification 
trainDF[,outcomeName] <- ifelse(trainDF[,outcomeName]==1,'yes','nope')

# create caret trainControl object to control the number of cross-validations performed
objControl <- trainControl(method='cv', number=2, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

# shuffling with GBM
objGBM <- train(trainDF[,predictorNames],  as.factor(trainDF[,outcomeName]),
                method='gbm',
                trControl=objControl,
                metric = "ROC",
                tuneGrid = expand.grid(n.trees = 5, interaction.depth = 3, shrinkage = 0.1)
)

predictions <- predict(object=objGBM, testDF[,predictorNames], type='prob')

GetROC_AUC = function(probs, true_Y){
        # AUC approximation
        # http://stackoverflow.com/questions/4903092/calculate-auc-in-r
        # ty AGS
        probsSort = sort(probs, decreasing = TRUE, index.return = TRUE)
        val = unlist(probsSort$x)
        idx = unlist(probsSort$ix) 

        roc_y = true_Y[idx];
        stack_x = cumsum(roc_y == 0)/sum(roc_y == 0)
        stack_y = cumsum(roc_y == 1)/sum(roc_y == 1)   

        auc = sum((stack_x[2:length(roc_y)]-stack_x[1:length(roc_y)-1])*stack_y[2:length(roc_y)])
        return(auc)
}

refAUC <- GetROC_AUC(predictions[[2]],testDF[,outcomeName])
print(paste('AUC score:', refAUC))

# Shuffle predictions for variable importance
AUCShuffle <- NULL
shuffletimes <- 500

featuresMeanAUCs <- c()
for (feature in predictorNames) {
        featureAUCs <- c()
        shuffledData <- testDF[,predictorNames]
        for (iter in 1:shuffletimes) {
                shuffledData[,feature] <- sample(shuffledData[,feature], length(shuffledData[,feature]))
                predictions <- predict(object=objGBM, shuffledData[,predictorNames], type='prob')
               featureAUCs <- c(featureAUCs,GetROC_AUC(predictions[[2]], testDF[,outcomeName]))
        }
        featuresMeanAUCs <- c(featuresMeanAUCs, mean(featureAUCs < refAUC))
}
AUCShuffle <- data.frame('feature'=predictorNames, 'importance'=featuresMeanAUCs)
AUCShuffle <- AUCShuffle[order(AUCShuffle$importance, decreasing=TRUE),]
print(AUCShuffle)

# shuffling with GLM
# change a few things for a linear model
objControl <- trainControl(method='cv', number=2)
trainDF[,outcomeName] <- ifelse(trainDF[,outcomeName]=='yes', 1, 0)

# GLM
objGLM <- train(trainDF[,predictorNames],  trainDF[,outcomeName],
                method='glm',
                trControl=objControl,
                preProc = c("center", "scale"))

predictions <- predict(object=objGLM, testDF[,predictorNames])
refRMSE=sqrt((sum((testDF[,outcomeName]-predictions[[2]])^2))/nrow(testDF))

VariableImportanceShuffle <- NULL

print(paste('Reference RMSE:',refRMSE))

shuffletimes <- 500
featuresMeanRMSEs <- c()
for (feature in predictorNames) {
     featureRMSEs <- c()
     shuffledData <- testDF[,predictorNames]
     for (iter in 1:shuffletimes) {
          shuffledData[,feature] <- sample(shuffledData[,feature], length(shuffledData[,feature]))
          predictions <- predict(object=objGLM, shuffledData[,predictorNames])
          featureRMSEs <- c(featureRMSEs, sqrt((sum((testDF[,outcomeName]-predictions[[2]])^2))/nrow(testDF)))
     }
     featuresMeanRMSEs <- c(featuresMeanRMSEs,  mean((featureRMSEs - refRMSE)/refRMSE))
}
VariableImportanceShuffle <- data.frame('feature'=predictorNames, 'RMSE_Importance'=featuresMeanRMSEs)
VariableImportanceShuffle <- VariableImportanceShuffle[order(VariableImportanceShuffle$RMSE_Importance),]
print(VariableImportanceShuffle)


# bonus - great package for fast variable importance
library(mRMRe)
ind <- sapply(titanicDF, is.integer)
titanicDF[ind] <- lapply(titanicDF[ind], as.numeric)
dd <- mRMR.data(data = titanicDF)
feats <- mRMR.classic(data = dd, target_indices = c(ncol(titanicDF)), feature_count = 10)
bestVars <-data.frame('features'=names(titanicDF)[solutions(feats)[[1]]], 'scores'= scores(feats)[[1]])
print(bestVars)