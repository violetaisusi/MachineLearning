main <- function(){
 
 #Load Packages
 library(RWeka)
 library(RWekajars)
 library(rJava)
 library(stringr)
 
 #Initialize vars for implementing randomForest Algorithm
 Number_of_Trees <- 500;Subspaces <- 10;treesPerSub <- 50;Feature_percent <- 0.75;runs <- 10
 #Initialize vars for calculating Average of Accuracy, AUC and Diversity per run. e.g totalAUC/runs
 totalAUC <- 0.0;totalEntropy <- 0.0;totalAccuracy <- 0.0
 
 #Initialize names of datasets, paths and names where datasets for training and testing are, and paths and names where datasets of subspaces will be stored
 nmds <- c("segment")
 pathdsTR <- c("arff-files/ds-ex-tr/segmentTR.arff")
 pathdsTS <- c("arff-files/ds-ex-ts/segmentTS.arff") 
 pathdssubTR <- c("arff-files/ds-tr-sub/sub1tr.arff","arff-files/ds-tr-sub/sub2tr.arff","arff-files/ds-tr-sub/sub3tr.arff","arff-files/ds-tr-sub/sub4tr.arff","arff-files/ds-tr-sub/sub5tr.arff","arff-files/ds-tr-sub/sub6tr.arff","arff-files/ds-tr-sub/sub7tr.arff","arff-files/ds-tr-sub/sub8tr.arff","arff-files/ds-tr-sub/sub9tr.arff","arff-files/ds-tr-sub/sub10tr.arff")
 pathdssubTS <- c("arff-files/ds-ts-sub/sub1ts.arff","arff-files/ds-ts-sub/sub2ts.arff","arff-files/ds-ts-sub/sub3ts.arff","arff-files/ds-ts-sub/sub4ts.arff","arff-files/ds-ts-sub/sub5ts.arff","arff-files/ds-ts-sub/sub6ts.arff","arff-files/ds-ts-sub/sub7ts.arff","arff-files/ds-ts-sub/sub8ts.arff","arff-files/ds-ts-sub/sub9ts.arff","arff-files/ds-ts-sub/sub10ts.arff")
 #Initialize path and name of dataset used for sampling subspacesdataset for training, it is needed for building the model
 pathsubsample <- "arff-files/ds-tr-sample/subsample.arff"
 pathOutputs <- "outputs/"
 
 lstClass <- vector(mode="character",length=length(nmds))#check after using only one of lstClassAtt or lstClasAttE . May be we can delete if calculations are correct
 
  #loop datasets - 15 datasets for training and 15 for testing
  for(dsCount in 1:length(nmds)){        
	 
    print(paste("Dataset",nmds[dsCount],sep=" -> "))
    
    ##writing output in a file intead of in consola.    
    pathOutput <- paste(pathOutputs,nmds[dsCount],".txt",sep="")    
    F_fileCreate(pathOutput)   
    
    dfOutput <- data.frame(info=paste("Dataset",nmds[dsCount],date(),sep=" - "))  
    ##
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
    
	  #loop runs for each dataset
    for(runCount in 1:runs){              
        AUC <- 0.0;totalWeight <- 0.0;correctlyClassified <- 0        
        #we build the matrix now to use it after to calculate votingweight
        actualEqualPredicted <- matrix(0,nrow=Number_of_Trees,ncol=numLinesDataTS)  
        classCountsE <- matrix(0,nrow=numLinesDataTS,ncol=numAttClassTR)#to calculate diversity per run after    
        size <- as.integer((Feature_percent * numAttTR))#in JAVA -1 xq indices empiezan en 0 xo como en R empiezan en 1 no restamos 1                          
        posClass <- size + 1 #position of attribute class to predict
        weight <- vector(mode="numeric",length=Subspaces)                      
        
	      for (subspaceCount in 1:Subspaces){			             	          
	          F_buildSubSpFiles(subspaceCount,nmds[dsCount],pathdssubTS,pathdssubTR,numAttTR,size,dfTRTableA,numLinesDataTR,numLinesDataTS,dfTSTableD,dfTR)          
	      }#end for subspaces 
        
        subspaceCount <- 0;limitTrees <- Number_of_Trees-1
        for(s in 0:limitTrees){#Number_of_Trees 500
          #treesPerSub 50          
          if(((s%%treesPerSub)==0) & (subspaceCount<Subspaces)){#treesPerSub change #even position in java because index starts in 0 - odd position in R because index of applicationArgs starts in 1      		                          
                      
            ##READ TRAINING SUBSPACES##
            subspaceCount <- subspaceCount + 1
            
            dfIGAE <- read.arff(pathdssubTR[subspaceCount])		   
            
            numAttSubTR <- ncol(dfIGAE)                          
            
            #InfoGainAttributeEval(descriptionModel,data=dataframe with data section) return a numeric vector with the figures of merit for the attributes specified by the right handside of formula
            IGAE <- InfoGainAttributeEval(class ~ .,data=dfIGAE)	
            dfSubTRTableA <- F_readDS(pathdssubTR[subspaceCount],FALSE,"\n","")                       
            classCounts <- vector(mode="integer",length=numAttClassTR) #In order to calculate entropy and weight[subsp]. 
            
            numLinesTableASubTR <- nrow(dfSubTRTableA)
            lineIniData <- grep('@data',dfSubTRTableA[1:nrow(dfSubTRTableA),])   
            
            ini <- lineIniData + 1
            numLinesDataSubTR <- numLinesTableASubTR - lineIniData  
            
            sampleVector <- c(ini:numLinesTableASubTR)
            probabilities <- rep(1/numLinesDataSubTR,times=numLinesDataSubTR)              
                    
            #sample with replacement with equal probability for each row (1/numLinesTableASubTR)
            #randomSample <- sample(sampleVector,numLinesDataSubTR,replace=TRUE,prob=probabilities)
            randomSample <- sample(sampleVector,numLinesDataSubTR,replace=TRUE,prob=probabilities)
            #print(sort.int(randomSample))
           
            rowsVector <- c(1:lineIniData,randomSample)                        
            
            dfSample <- data.frame(dfSubTRTableA[rowsVector,])  
                                  
            F_fileCreate(pathsubsample)                        
            F_writeTable(dfSample ,pathsubsample ,"\n",TRUE,FALSE,FALSE,FALSE) 
            
            ##READ TEST SUBSPACES in oder to evaluate the model after                                    
            dfSubTS <- read.arff(pathdssubTS[subspaceCount])                                                              
            
            ##CALCULATE ENTROPY FOR WEIGHTS[]                                       
            
            for(l in ini:numLinesTableASubTR){ #loop all lines of data section                
              #find the last att (because are ordered so the att of class to predict will be in last position)                     
              v1 <- unlist(strsplit(as.character(dfSubTRTableA[l,]),",",fixed=TRUE))	               
              #if the last att en the line is att of the class to predict, plus 1 in index of classCounts                    
              pos <- match(c(as.character(v1[length(v1)])),noquote(lstClass[[dsCount]]),nomatch=0,incomparables=0)                     
              if(pos > 0)
                classCounts[pos] <-  classCounts[pos] + 1                               
            }                           
            
            tmp <- 0
            for(j in 1:numAttClassTR){#
              if(classCounts[j] > 0){                
                tmp <- tmp - (classCounts[j] * log2(classCounts[j]))        
              }
            }                        
            tmp <- tmp / numLinesDataTR            
            entropy <- tmp + log2(numLinesDataTR)#after we are going to use this value of entropy to modify the vector of weight                  
                       
            IGAE[is.na(IGAE)] <- 0 #if there is some NA in IGAE vector we replace it by 0
            
            tmp <- 0
            for(j in 1:length(IGAE)){#length(IGAE) is always one position shorter than numAttSubTR because attributes predict class that is the last		  
              tmp <- tmp + (IGAE[j] / entropy)                 
            }                        
            weight[subspaceCount] <- as.double(1/numAttSubTR) * tmp #/by length(IGAE) rather /numAttClassTR because sometimes IGAE[] returns NA                                                   
            totalWeight <- totalWeight + (treesPerSub * weight[subspaceCount])#totalWeight for run                                    
          }#end if treesPerSub                       
                              		                              
          #I build the model of classifier RandomTree with training subspace dataset sampled
          RT <- make_Weka_classifier("weka/classifiers/trees/RandomTree")	                                                  
          
          model <- RT(class ~ .,data=read.arff(pathsubsample))#All datasets have the same name for class to predict: class ~ .                    
          #Predictions with testing subspace dataset
          predictions <- predict(model,newdata=dfSubTS)				          
          #I evaluate the model with test data
          evaluatedModel <- evaluate_Weka_classifier(model,newdata=dfSubTS,complexity=TRUE,class=TRUE)                   	                     
          #with class=TRUE we can have stadistics such as AUC
          detailsClass <- evaluatedModel$detailsClass
          
          #nrow(detailsClass) one row for each different value in class
          #ncol 6 is nrow(detailsClass) for each different value in class
          #So we need to calculate weightedAreaUnderRoc
          tmpAUC <- 0.0
          weightedAUC <- 0.0
          for(p in 1:nrow(detailsClass)){            
            if(!is.nan(detailsClass[p,6]))
              tmpAUC <- tmpAUC + detailsClass[p,6]
          }
          weightedAUC <- tmpAUC / nrow(detailsClass)                        
          AUC <- AUC + weightedAUC           
                    
          #TS - prepare info for calculate entropy and vector of entropies                                        
          actual <- ""
          
          #numLinesDataTS is 1 position bigger than length(predictions), maybe for headers
          for(j in 1:(length(predictions))){		                           
                              
            dfSubTSTableD <- F_readDS(pathdssubTS[subspaceCount],FALSE,",","@")
            #value of class attribute in SubTesti.arff            
            #if the last att en the line is att of the class to predict add 1 in index of classCounts                            
            pos <- match(c(as.character(dfSubTSTableD[j,posClass])),noquote(lstClass[[dsCount]]),nomatch=0,incomparables=0)              
            if(pos > 0){
              actual <- as.character(dfSubTSTableD[j,posClass])              
            }
            
            predicted <- predictions[j] 
                       
            #(s+1) because index for number_of_tress starts in 0
            if(actual==predicted){             
              actualEqualPredicted[(s+1),j] <- 1                   
            }   
                                                   
            #plus 1 in index of classCountsE for predicted classes            
            posE <- match(c(as.character(predicted)),noquote(lstClass[[dsCount]]),nomatch=0,incomparables=0)  
            
            if(posE > 0)
              classCountsE[j,posE] <- classCountsE[j,posE] + 1                                                             
            
          } #end length(predictions)                  
          ##
          #entropies
          #remove values 0 to improve loop            
          #classCountsE <- classCountsE[classCountsE > 0]                            
          ##                          
          
        }#for s Number_of_Trees                   
        
        entropies <- vector(mode="numeric",length=length(predictions))     
        for(l in 1:length(predictions)){#length(predictions) is 1 row less than numLinesDataTS
          
          votingWeightSub <- vector(mode="numeric",length=Subspaces)#initializes a list of Subspaces number of elements with value 0
          for(i in 1:Subspaces){            
            for(s in 1:Number_of_Trees){                                     
              #votingweightSub
              if(actualEqualPredicted[s,l]==1){                
                votingWeightSub[i] <- votingWeightSub[i] + weight[i]                
              }
            }#for Number_of_Trees            
          }#for Subspaces          
          if(votingWeightSub[i] > (totalWeight/2.0))
            correctlyClassified <- correctlyClassified + 1
          
          #calculations for entropies
          tmp <- 0
          for(c in 1:ncol(classCountsE)){#ncol(classCounts) is the number of possible value in class attribute            
            if(classCountsE[l,c] > 0){#when classCountsE[r] is 0 because this attribute hasn't any occurrency, log2(classCountsE[r]) is -inf and * results NAN
              tmp <- tmp - (classCountsE[l,c] * log2(classCountsE[l,c]))             
            }
          }
          
          tmp <- tmp / Number_of_Trees #numLinesDataE is equal to number of trees
          entropy <- tmp + log2(Number_of_Trees)#numLinesDataE is equal to number of trees
          
          entropy <- entropy / log2(ncol(classCountsE));                    
          
          entropies[l] <- entropy  
          
        }#for numLinesDataTS 
        print(paste("Run", runCount,date(), sep="  "))
        
        infoOuput<- data.frame(info=paste("Run", runCount,date(), sep="  "))
        dfOutput <- rbind(dfOutput,infoOuput)
        
        infoOuput<- data.frame(info=paste("correctly Classified for this run",correctlyClassified,"of",length(predictions),sep=" -> ")) 
        dfOutput <- rbind(dfOutput,infoOuput)
        
        totalAccuracy <- totalAccuracy + (correctlyClassified/length(predictions))        
                
        AvgAccuracySub <- correctlyClassified / length(predictions)
        
        infoOuput<- data.frame(info=paste("correctly Classified Rate -> Accuracy rate for this run ->", AvgAccuracySub * 100, "%",sep=" "))
        dfOutput <- rbind(dfOutput,infoOuput)
        
        AvgAUCRun <- (AUC/Number_of_Trees)
        
        infoOuput<- data.frame(info=paste("AUC rate for this run ->", AvgAUCRun * 100,"%", sep=" "))
        dfOutput <- rbind(dfOutput,infoOuput)
        
        totalAUC <- totalAUC + AvgAUCRun
        
        infoOuput<- data.frame(info=paste("Entropy - Diversity rate for this run ->", mean(entropies) * 100,"%", sep=" -> "))
        dfOutput <- rbind(dfOutput,infoOuput)
        
        totalEntropy <- (totalEntropy + mean(entropies))        
                
    }#end for runcount      	  
    
    infoOuput<- data.frame(info=paste("Average Accuracy ->", (totalAccuracy/runCount) * 100,"%", sep=" "))
    dfOutput <- rbind(dfOutput,infoOuput)
    
    infoOuput<- data.frame(info=paste("Average AUC ->", (totalAUC/runCount)*100,"%", sep=" "))
    dfOutput <- rbind(dfOutput,infoOuput)
    
    infoOuput<- data.frame(info=paste("Average Entropy - Diversity ->", (totalEntropy/runCount) * 100,"%", sep=" "))
    dfOutput <- rbind(dfOutput,infoOuput)
    
    F_writeTable(dfOutput ,pathOutput ,"\n",TRUE,FALSE,FALSE,FALSE) 
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

F_buildSubSpFiles <- function(subspaceCount,nmds,pathdssubTS,pathdssubTR,numAttTR,size,dfTRTableA,numLinesDataTR,numLinesDataTS,dfTSTableD,dfTR){
  
    dfSubTR <- data.frame()#dataframe to store info of training subspaces files before write in file.
    dfSubTS <- data.frame()#dataframe to store info of testing subspaces files before write in file.	       
    
    #CREATE FILES subspaces
    #TESTING - Until data lines is equal to training dataset
    nameFileSubSpaceTS <- pathdssubTS[subspaceCount]	          
    F_fileCreate(nameFileSubSpaceTS)	
    
    #TRAINING	         	
    nameFileSubSpaceTR <- pathdssubTR[subspaceCount]	          
    F_fileCreate(nameFileSubSpaceTR)			               
    infoFileSubSpaceTR <- data.frame(info=paste("@relation ",nmds,sep="")  )           
    dfSubTR <- rbind(dfSubTR,infoFileSubSpaceTR)			      		      			            		        		                     
    
    #generate random vector of number in range (1 - number of attibutes of dataset) without any number repeteated		        
    #False because is without replacement
    ar <- sample.int((numAttTR-1),size,FALSE,NULL)# -1 because the last attribute is the class                                    		                               
    #order random vector        
    ar <- sort.int(ar,partial=NULL,na.last=NA,decreasing=FALSE,method=c("quick"))                                 		
    
    #write in subspaces files: names and values of attributes that for position are in random vector of each dataset
    #it's the same random vector for TR and TS                   
    for (k in 1:size){#size = length(ar)		               
      dfSubTR <- rbind(dfSubTR,data.frame(info=as.character(dfTRTableA[ar[k],])))#Sub training              
    }#end for k        		       		            
    
    #write the class attribute (in the last position)			      
    dfSubTR <- rbind(dfSubTR,data.frame(info=as.character(dfTRTableA[numAttTR,])))#Sub training			      			    
    #write data header line		        
    dfSubTR <- rbind(dfSubTR,data.frame(info="@data"))#Sub training		  
    
    dfSubTS <- rbind(dfSubTS,dfSubTR)#Until here the same for Sub Testing		          		        		            
    
    #writing in subspaces TR and TS data values of attributes that are in random vector			      
    beforeLastLineTR <- (numLinesDataTR - 1);beforeLastLineTS <- (numLinesDataTS - 1)
    #Given that always testing files have less lines of data than training files
    #in order to improve the performance of R code I join together tasks for training and testing data sample
    for(l in 1:beforeLastLineTR){
      str <- "";strts <- "";k <- 1            
      for(k in 1:length(ar)){				        					       			       
        if(l <= beforeLastLineTS)
          strts <- paste(strts,dfTSTableD[l,ar[k]],",",sep="") 
        str <- paste(str,dfTR[l,ar[k]],",",sep="")           			    				      
      }#end for j
      #Concat the last attribute			        
      if(l <= beforeLastLineTS){			          			          
        dfSubTS <- rbind(dfSubTS,data.frame(info=paste(strts,dfTSTableD[l,numAttTR],sep=""))  )  
      }			          
      dfSubTR <- rbind(dfSubTR,data.frame(info=paste(str,dfTR[l,numAttTR],sep="")))			 				        
    }#end for l		        
    
    #write to file TR and TS the info			      	
    F_writeTable(dfSubTR ,nameFileSubSpaceTR ,"\n",TRUE,FALSE,FALSE,FALSE) 
    F_writeTable(dfSubTS,nameFileSubSpaceTS ,"\n",TRUE,FALSE,FALSE,FALSE)  	
}