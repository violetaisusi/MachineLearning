main <- function(){
 
 #Load Packages
 library(RWeka)
 library(RWekajars)
 library(rJava)
 library(stringr)
 
 #Initialize vars for implementing randomForest Algorithm
 Number_of_Trees <- 500;runs <- 1
 #Initialize vars for calculating Average of Accuracy, AUC and Diversity per run. e.g totalAUC/runs
 totalAUC <- 0.0;totalEntropy <- 0.0;totalAccuracy <- 0.0; totalOverallAccuracyRuns <- 0.0
 
 #votingWeightSub <- rep.int(1,Subspaces)#initilized to 1s because after we are going to multiply for weight[i] 
 #Initialize names of datasets, paths and names where datasets for training and testing are, and paths and names where datasets of subspaces will be stored
 nmds <- c("iris")
 pathdsTR <- c("arff-files/ds-tr/irisTR.arff")
 pathdsTS <- c("arff-files/ds-ts/irisTS.arff") 
 
 lstClass <- vector(mode="character",length=length(nmds))#check after using only one of lstClassAtt or lstClasAttE . May be we can delete if calculations are correct
 
  #loop datasets - 15 datasets for training and 15 for testing
  for(dsCount in 1:length(nmds)){        
	 
    print(paste("Dataset",nmds[dsCount],sep=" -> "))
	  #READ FILES   
  	#training
    dfTRTableA <- F_readDS(pathdsTR[dsCount],TRUE,"\n","")
    dfTR <- F_readDS(pathdsTR[dsCount],FALSE,",","@")
        
    lineDataTR <- grep('@data',dfTRTableA[1:nrow(dfTRTableA),])
	  numLinesDataTR <- nrow(dfTRTableA) - lineDataTR  	  	
    numAttTR <- ncol(dfTR)#ncol is number of attributes   
    
    #vector with different attributes of class to predict in order to calculate entropy after	  
	  v1 <- unlist(strsplit(as.character(dfTRTableA[numAttTR,]),"{",fixed=TRUE))	 
	  v2 <- unlist(strsplit(v1[2],"}",fixed=TRUE))	  	      
	  lstClass[dsCount] <- as.vector(strsplit(v2[1],",",fixed=TRUE))
	  numAttClassTR <- length(lstClass[[dsCount]])            
        
	  #testing
    dfTSTableA <- F_readDS(pathdsTS[dsCount],TRUE,"\n","")
    dfTSTableD <- F_readDS(pathdsTS[dsCount],FALSE,",","@")

	  numLinesDataTS <- nrow(dfTSTableD)   	
  	#numAttTS <- ncol(dfTSTableD)  	num attributes in testing files is equal to num attributes in training files
    
	  #no loop for runs in RF original. We only run the algorithm ones           
        AUC <- 0.0;correctlyClassified <- 0;cont <- 0              
        classCountsE <- matrix(0,nrow=numLinesDataTS,ncol=numAttClassTR)#to calculate diversity per run after                                              	           
                              		                              
          #I build the model of classifier RandomTree with training subspace dataset sampled
          RT <- make_Weka_classifier("weka/classifiers/trees/RandomForest")                    
          model <- RT(class ~ .,data=read.arff(pathdsTR[dsCount]))         
          #Predictions with testing subspace dataset
          predictions <- predict(model,newdata=read.arff(pathdsTS[dsCount]),type=c("class"))				          
          #I evaluate the model with test data
          #evaluatedModel <- evaluate_Weka_classifier(model,newdata=read.arff(pathdsTS[dsCount]),complexity=TRUE,class=TRUE)                   	                     
          evaluatedModel <- evaluate_Weka_classifier(model,newdata=read.arff(pathdsTS[dsCount]),class=TRUE)                                                
          #with class=TRUE we can have stadistics such as AUC
          #detailsComplexity <- evaluatedModel$detailsComplexity #return a vector
                   
          detailsClass <- evaluatedModel$detailsClass
          #print(paste("colnames(detailsClass)",colnames(detailsClass),sep="->"))
          #detailsClass: [1] falsePositiveRate(FP) 
          #              [2] falseNegativeRate(FN)
          #              [3] precision (Accuracy per class)
          #              [4] recall
          #              [5] fMeasure
          #              [6] areaUnderROC
         
          #nrow(detailsClass) one row for each different value in class
          #ncol 6 is nrow(detailsClass) for each different value in class
          #So we need to calculate weightedAreaUnderRoc
          tmpAUC <- 0.0;weightedAUC <- 0.0;overallAccuracyModel <- 0.0
                  
          for(p in 1:nrow(detailsClass)){                       
              AUC <- AUC + detailsClass[p,6]             
            overallAccuracyModel <- overallAccuracyModel + detailsClass[p,3]            
          }
                                 
          AUC <- AUC / nrow(detailsClass)
          overallAccuracyModel <- overallAccuracyModel / nrow(detailsClass)          
          
          #TS - prepare info for calculate entropy and vector of entropies                                        
          actual <- "";
          actuals <- vector(mode="numeric",length=numAttClassTR)
          
          #numLinesDataTS is 1 position bigger than length(predictions), maybe for headers
          for(j in 1:(length(predictions))){		                           
                              
            dfSubTSTableD <- F_readDS(pathdsTS[dsCount],FALSE,",","@")
            #value of class attribute in SubTesti.arff            
            #if the last att en the line is att of the class to predict add 1 in index of classCounts   
            
            pos <- match(c(as.character(dfSubTSTableD[j,ncol(dfSubTSTableD)])),noquote(lstClass[[dsCount]]),nomatch=0,incomparables=0)              
            if(pos > 0){
              actual <- as.character(dfSubTSTableD[j,ncol(dfSubTSTableD)])   
              actuals[pos] <- actuals[pos] + 1
            }
            
            predicted <- predictions[j]                         
            
            if(actual==predicted) cont<-cont+1
                                                   
            #plus 1 in index of classCountsE for predicted classes            
            posE <- match(c(as.character(predicted)),noquote(lstClass[[dsCount]]),nomatch=0,incomparables=0)  
            
            if(posE > 0)
              classCountsE[j,posE] <- classCountsE[j,posE] + 1                                                             
            
          } #end length(predictions)                          
                
        #calculations for entropies
        tmp <- 0;entropy <- 0
        for(c in 1:ncol(classCountsE)){#ncol(classCounts) is the number of possible value in class attribute
          sumAttr <- sum(classCountsE[,c])          
          attr <- (sumAttr/length(predictions))
          
          if(attr > 0){#when classCountsE[r] is 0 because this attribute hasn't any occurrency, log2(classCountsE[r]) is -inf and * results NAN
            tmp <- -(attr * log2(attr))               
            entropy <- entropy + (attr * tmp)            
          }
        }               
                
        print(paste("entropy", entropy, sep=" -> "))                                  
        
        print(paste("overallAccuracyModel rate for this run",overallAccuracyModel * 100,"%",sep=" "))
        
        print(paste("correctly Classified for this run",cont,"of",length(predictions),sep=" -> "))                           
        
        correctlyClassifiedRate <- (cont / length(predictions)) * 100
        print(paste("correctly Classified Rate -> Accuracy rate for this run ->", correctlyClassifiedRate, "%",sep=" "))
        
        totalAccuracy <- totalAccuracy + correctlyClassifiedRate 
        
        totalOverallAccuracyRuns <- totalOverallAccuracyRuns + overallAccuracyModel                      
                                      
        totalAUC <- totalAUC + AUC
        print(paste("AUC rate for this run ->",AUC * 100,"%", sep=" "))     
        print(paste("Entropy - Diversity ->", entropy, sep=" "))        
        totalEntropy <- (totalEntropy + entropy)            	  	  	   
	  
  }#end for datasetNo
 
}
#if sep="\n" all lines in the file 
#if sep="," & comments="@") only data lines in the fils with one column for each attribute
F_readDS <- function(pathnmds,hd,sep,comments) {
  return(read.table(pathnmds, header = hd, sep = sep, quote = "\"'",comment.char = comments))
}
F_fileCreate <- function(pathFile){
  file.create(pathFile)
}
F_writeTable <- function(df,pathFile,endOfLine,apd,qts,rNames,cNames){
  write.table(df,pathFile ,eol=endOfLine,append=apd,quote=qts,row.names=rNames,col.names=cNames)              
}