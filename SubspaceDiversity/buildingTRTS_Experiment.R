main <- function(){
 
 #Load Packages
 library(RWeka)
 library(RWekajars)
 library(rJava)
 library(stringr)
  
 #Initialize names of datasets, paths and names where datasets for training and testing are, and paths and names where datasets of subspaces will be stored

 nmds <- c("arff-files/ds-ex/balance-scale.arff","arff-files/ds-ex/kr-vs-kp.arff","arff-files/ds-ex/glass.arff","arff-files/ds-ex/segment.arff")
 pathdsTR <- c("arff-files/ds-ex-tr/balance-scaleTR.arff","arff-files/ds-ex-tr/kr-vs-kpTR.arff","arff-files/ds-ex-tr/glassTR.arff","arff-files/ds-ex-tr/segmentTR.arff")
 pathdsTS <- c("arff-files/ds-ex-ts/balance-scaleTS.arff","arff-files/ds-ex-ts/kr-vs-kpTS.arff","arff-files/ds-ex-ts/glassTS.arff","arff-files/ds-ex-ts/segmentTS.arff")
 pathRandomizedDS <- "arff-files/ds-tr-sample/randomizedDS.arff"
 
  #loop datasets
  for(dsCount in 1:length(nmds)){        
	 
    print(paste("Dataset",nmds[dsCount],sep=" -> "))
	  #READ dataset   	
    dfTable <- F_readDS(nmds[dsCount],FALSE,"\n","")#for lines
    df <- F_readDS(nmds[dsCount],FALSE,",","@")#for attribute values
    
    
    ###############RANDOMIZED DS######################
    
    lineData <- grep('@data',dfTable[1:nrow(dfTable),])
    ini <- lineData + 1
	  numLinesData <- nrow(dfTable) - lineData  	  	
    numAtt <- ncol(df)#ncol is number of attributes      
    
    linesVector <- c(ini:nrow(dfTable))
    probabilities <- rep(1/numLinesData,times=numLinesData)              
            
    #sample without replacement with equal probability for each row (1/numLinesTableASubTR)    
    randomSample <- sample(linesVector,numLinesData,replace=FALSE,prob=probabilities)
    #print(sort.int(randomSample))
    
    rowsVector <- c(1:lineData,randomSample)                        
    
    dfRamdomized <- data.frame(dfTable[rowsVector,])  
    
    F_fileCreate(pathRandomizedDS)                        
    F_writeTable(dfRamdomized ,pathRandomizedDS ,"\n",TRUE,FALSE,FALSE,FALSE) 
    
    #########################
    
    dfTable <- F_readDS(pathRandomizedDS,FALSE,"\n","")#for lines
    df <- F_readDS(pathRandomizedDS,FALSE,",","@")#for attribute values            
    
    lineData <- grep('@data',dfTable[1:nrow(dfTable),])
    ini <- lineData + 1
    numLinesData <- nrow(dfTable) - lineData  	  	
    numAtt <- ncol(df)#ncol is number of attributes  
    
    
    numLinesDataTR <- (numLinesData * (2/3))
    numLinesDataTS <- (numLinesDataTR * (1/3))     
    
    dfTR <- data.frame()#dataframe to store info of training subspaces files before write in file.
    dfTS <- data.frame()#dataframe to store info of testing subspaces files before write in file.         
    
    #CREATE FILES
    #TESTING - Until data lines is equal to training dataset          
    F_fileCreate(pathdsTS[dsCount])	        
    
    #TRAINING	         	    	          
    F_fileCreate(pathdsTR[dsCount])			               
        
    lastTRline <- ini + numLinesDataTR 
    rowsVectorTR <- c(1:lineData,ini:lastTRline)  
    #print(rowsVectorTR)
    
    infoFileTR <- data.frame(info=as.character(dfTable[rowsVectorTR,])) 
    dfTR <- rbind(dfTR,infoFileTR) 
    
    firstTSline <- lastTRline + 1 
    rowsVectorTS <- c(1:lineData,firstTSline:nrow(dfTable))
    #print(rowsVectorTS)
    
    infoFileTS <- data.frame(info=as.character(dfTable[rowsVectorTS,])) 
    dfTS <- rbind(dfTS,infoFileTS) 
    
    #write to file TR and TS the info  		      	
    F_writeTable(dfTR ,pathdsTR[dsCount] ,"\n",TRUE,FALSE,FALSE,FALSE) 
    F_writeTable(dfTS,pathdsTS[dsCount] ,"\n",TRUE,FALSE,FALSE,FALSE)  	    	
	     	  
  }#end for datasetNo
 
}#end main

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

F_checkOverlap <- function(dfTR,strts){
  
  for(i in 1:nrow(dfTR)){    
    if(as.character(dfTR[i,])==as.character(strts)){
      #print(paste("i",i,as.character(dfTR[i,]),"strts",strts,sep="->"))
      return (TRUE)
    }
  }
  return (FALSE)
    
}