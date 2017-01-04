/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package subspacediversity;

import java.io.*;

import weka.core.Utils;

import java.util.*;

import weka.core.Instances;

import weka.core.Instance;

import weka.classifiers.Evaluation;

import weka.core.converters.ArffLoader;

import weka.classifiers.trees.RandomTree;

import weka.attributeSelection.InfoGainAttributeEval;

import java.io.FileWriter;

import java.io.BufferedWriter;
/**
 *
 * @author kfawagreh
 */
public class SubspaceDiversity {

    /**
     * @param args the command line arguments
     */
    
    public static final int Number_of_Trees = 500;
    
    public static final int Subspaces = 10;
    
    public static final int treesPerSub = 50;
    
    public static final double Feature_percent = 0.75;
     
    public static void main(String[] args) {
        // TODO code application logic here
        
       
        
        try {
     
        int datasetName=70,entropyCounter=0;
        Evaluation eval=null;
        double [] classCounts;
        double tmp;
        double[] entropies;
        String str= new String();
        
        
        
        for (int datasetNo=0; datasetNo < 30; datasetNo++)
        {
  
        
                
        Instances training=null,testing=null,TR=null,TS=null,DSEntropy=null,sample;
        
        ArffLoader loader=null;
        
        Instance inst;
        
        LinkedList<RandomTree> randomForest;
        double[] weight;
        double votingWeight=0;
        double AUC=0,totalAUC=0;
        double entropy=0,totalEntropy=0;
        double weightToUse;
        RandomTree rt;
        int i,j,k,temp,correctlyClassified=0,runCount=0,randomNum,instCounter=0;
        String actual="", predicted="", wronglbl="";
        Iterator iterator;
        double total=0;
        long seed=1000000;
        Random rm;
        BufferedWriter writer=null,writer2=null,writer3=null;
        rm = new Random();
        boolean light=true;
        InfoGainAttributeEval IGAE = new InfoGainAttributeEval();
        System.out.println("Dataset=" + args[datasetName]);
        loader = new ArffLoader();
        loader.setFile(new File(args[datasetNo++]));
        training = loader.getDataSet();
        training.setClassIndex(training.numAttributes() - 1 );
        
        loader.setFile(new File(args[datasetNo]));
        testing = loader.getDataSet();
        testing.setClassIndex(testing.numAttributes() - 1 );
        
        writer3 = new BufferedWriter(new FileWriter("DSForEntropy.arff"));
        writer3.write("@relation " + args[datasetName]);
        writer3.newLine();
        str="";
        for (runCount=0; runCount < 10; runCount++)
        {
        k=0;   
        AUC=0;
        entropy=0;
        
        int size = (int)(Feature_percent*(training.numAttributes()))-1;
        
        int[] ar = new int[size];
        
        
        for (int subspaceCount=0; subspaceCount < Subspaces; subspaceCount++)
        {
            writer = new BufferedWriter(new FileWriter("Sub"+subspaceCount+".arff"));
            writer.write("@relation " + args[datasetName]);
            writer2 = new BufferedWriter(new FileWriter("Sub"+subspaceCount+"Test.arff"));
            writer2.write("@relation " + args[datasetName]);
            
            writer.newLine();
            writer2.newLine();
            for (i=0; i < size; i++) {
                seed = rm.nextInt();
                rm.setSeed(seed);
                randomNum = rm.nextInt(training.numAttributes()-1);
                
           
                boolean found = false;
                for (k=0; k < i && !found; k++) {
                  if (ar[k]==randomNum) found=true;
              }
              if (!found) ar[i]=randomNum;
              else i--;
              
            }
            boolean swapped = true;
            j=0;
            while (swapped) {
            swapped = false;
             j++;   
            for (i = 0; i < ar.length - j; i++) {
                      if (ar[i] > ar[i + 1]) {
                         temp = ar[i];
                         ar[i] = ar[i + 1];
                         ar[i + 1] = temp;
                         swapped = true;
                      }
                 }
            }
     
           
           
            for (i = 0; i < ar.length; i++) {
            if (light==false)
               System.out.println("ar[" + i + "]=" + ar[i]);  
            writer.write(training.attribute(ar[i]).toString());
            writer2.write(training.attribute(ar[i]).toString());
            if ((subspaceCount==0) && (runCount==0)) {
            writer3.write(training.attribute(ar[i]).toString());
            writer3.newLine();
            }
            writer.newLine();
            writer2.newLine();
            
            }
        
        writer.write(training.attribute(training.numAttributes()-1).toString());
        writer.newLine();
        writer.write("@data ");
        writer2.write(training.attribute(training.numAttributes()-1).toString());
        writer2.newLine();
        writer2.write("@data ");
        if ((subspaceCount==0) && (runCount==0)) {
        writer3.write(training.attribute(training.numAttributes()-1).toString());
        writer3.newLine();
        writer3.write("@data ");
        }
        
        for (i = 0; i < training.size(); i++) {
          
            k=0;
            for (j = 0; j < training.numAttributes(); j++)
            {
                if (k==size) break;
                if (ar[k]==j)
                {
                   if ((subspaceCount==0) && (i==0))
                       str += training.instance(i).toString(j) + ",";
                   writer.write(training.instance(i).toString(j) + ",");
                   k++;
                }
            }
            if  ((subspaceCount==0) && (i==0))
                str += training.instance(i).toString(training.numAttributes()-1);
            writer.write(training.instance(i).toString(training.numAttributes()-1));
            writer.newLine();
            
        }
        writer.close();
        k=0;
        
        for (i = 0; i < testing.size(); i++) {
            k=0;
            for (j= 0; j < testing.instance(i).numAttributes(); j++)
            {
                if ((k < size) && (ar[k] == j))
                {
                   
                    writer2.write(testing.instance(i).toString(j) + ",");
                    k++;
                }
          
                
            }
            writer2.write(testing.instance(i).toString(testing.numAttributes()-1) + " ");
            writer2.newLine();
        }
        writer2.close();
        }
        if (runCount==0) {
        for (i=0; i < Number_of_Trees; i++)
        {
            writer3.write(str);
            writer3.newLine();
        }
        writer3.close();
        }
        
        int subspace=30;
        Instances[] instArrTS = new Instances[Subspaces];
        Instances[] instArrTR = new Instances[Subspaces];
        entropies = new double [testing.size()];
            randomForest = new LinkedList<RandomTree>();
            correctlyClassified=0;
            
            subspace=30;
            weight = new double[Subspaces];
            k=0;
            int t=0;
            for (i=0; i < Number_of_Trees; i++)
            {
                    if ((i%treesPerSub)==0)
                    {
                        loader = new ArffLoader();
                        loader.setFile(new File(args[subspace++]));
                        TR = loader.getDataSet();
                        TR.setClassIndex(TR.numAttributes() - 1 );
                        instArrTR[t++]=TR;
                        classCounts = new double[TR.numClasses()];
                        Enumeration instEnum = TR.enumerateInstances();
                        while (instEnum.hasMoreElements()) {
                         inst = (Instance) instEnum.nextElement();
                         classCounts[(int) inst.classValue()]++;
                         }
                
                         tmp = 0;
                         for (j = 0; j < TR.numClasses(); j++) {
                            if (classCounts[j] > 0) {
                              tmp -= classCounts[j] * Utils.log2(classCounts[j]);
                            }
                         }
                         tmp /= (double) TR.numInstances();
                         entropy = tmp + Utils.log2(TR.numInstances());
                         IGAE.buildEvaluator(TR);
                         tmp=0;
                        for (j=0; j < TR.numAttributes(); j++)
                        {
                           tmp = tmp + (IGAE.evaluateAttribute(j) / entropy);
                        }
                        weight[k++]= ((double)1/TR.numAttributes())*tmp;
                    }
                    sample = TR.resample(new Random());
                    sample.setClassIndex(TR.numAttributes() - 1 );
                    rt = new RandomTree();
                    rt.buildClassifier(sample);
                    randomForest.add(rt);
                    if ((light==false) && (i%treesPerSub)==0)
                        System.out.println("i=" + i + " hash=" + rt.hashCode());
                    
            } 
            double totalWeight = 0;
            for (k=0; k < Subspaces; k++)
                totalWeight += treesPerSub*weight[k];
            
            i=0;
            subspace=50;
            for (k=0; k < Subspaces; k++)
            {
             loader = new ArffLoader();
             loader.setFile(new File(args[subspace++]));
             TS = loader.getDataSet();
             TS.setClassIndex(TS.numAttributes() - 1 );
             instArrTS[i++]=TS;
            }
            t=0;
             int treeNo=-1;
             loader = new ArffLoader();
             loader.setFile(new File("DSForEntropy.arff"));
             DSEntropy = loader.getDataSet();
             DSEntropy.setClassIndex(DSEntropy.numAttributes() - 1 );
             
            iterator = randomForest.iterator();
            while (iterator.hasNext()) {
                    
                    rt = (RandomTree) iterator.next();
                    eval = new Evaluation(instArrTR[t]);
                    eval.evaluateModel(rt, instArrTS[t]);
                    treeNo++;
                    if (treeNo > 0 && (treeNo+1)%treesPerSub==0)
                        t++;
                    AUC += eval.weightedAreaUnderROC();
                }
                
            
            entropyCounter=0;
            for (j=0; j < instArrTS[0].size(); j++)
            {
                
                iterator = randomForest.iterator();
                
                 boolean newSub=false;
                 k=0;
                 
                 votingWeight=0;
                 instCounter = 0;
                 
            for (i=0; i <  instArrTS.length; i++)
            {
                
               
               weightToUse = weight[k++];
               actual= instArrTS[i].instance(j).toString(instArrTS[i].numAttributes() - 1);
                 
               treeNo=-1;
               
               
                while (iterator.hasNext()) {
                       
                      treeNo++;
                      
                      rt = (RandomTree) iterator.next();
                      if ((newSub) && (light==false)) {System.out.println("code=" + rt.hashCode());newSub=false;}
                      
                      double clsLabel = rt.classifyInstance(instArrTS[i].instance(j));
                      
                      DSEntropy.instance(instCounter++).setClassValue(clsLabel);
                      predicted = instArrTS[i].instance(j).classAttribute().value((int) clsLabel);
                     
                      if (light==false)
                        System.out.print(predicted+" ");
                      if (actual.equals(predicted))
                          votingWeight = votingWeight + weightToUse;
                      else wronglbl = predicted;
                      
                      if (((treeNo+1)%treesPerSub)==0)
                      {
                         newSub=true;
                         break;
                      }
                      }
            }
            
                        
                        
                        classCounts = new double[DSEntropy.numClasses()];
                        Enumeration instEnum = DSEntropy.enumerateInstances();
                        while (instEnum.hasMoreElements()) {
                         inst = (Instance) instEnum.nextElement();
                         classCounts[(int) inst.classValue()]++;
                        }
                        tmp = 0;
                        for (int x = 0; x < DSEntropy.numClasses(); x++) {
                            if (classCounts[x] > 0) {
                              tmp -= classCounts[x] * Utils.log2(classCounts[x]);
                            }
                         }
                         tmp /= (double) DSEntropy.numInstances();
                         entropy = tmp + Utils.log2(DSEntropy.numInstances());
                         entropy /= Utils.log2(DSEntropy.numClasses());
                         entropies[entropyCounter++]=entropy;   
                      
                      if (votingWeight > (totalWeight/2.0))
                      {
                          if (light==false)
                            System.out.println("Majority Voting->" + actual);
                          correctlyClassified++;
                      }
                      else
                      {
                         
                             if (light==false)
                               System.out.println("Majority voting->" + wronglbl + "(incorrect classsification)");
                             
                      }
                      
            
                } 
                System.out.println("Run#->" + (runCount+1));
                total = total + (double)correctlyClassified/instArrTS[0].size();
                System.out.printf("Accuracy Rate=%.5f%s\n" , (double)correctlyClassified/testing.size()*100,"%");
                System.out.printf("AUC=%.5f%s\n" , (AUC/randomForest.size())*100,"%");
                totalAUC  += AUC/randomForest.size();
                double sum=0;
                
                for (i=0; i < entropies.length; i++)
                    sum = sum + entropies[i];
              
                System.out.printf("Diversity=%.5f\n" , (sum/entropies.length));
                totalEntropy += sum/entropies.length;
        }
              
        
        System.out.println("RunCount->" + runCount); 
        System.out.printf("Average Accuracy=%.5f%s\n" , (total/runCount)*100,"%");
        System.out.printf("Average AUC=%.5f%s\n" , (totalAUC/runCount)*100,"%");
        System.out.printf("Average Diversity=%.5f\n" , (totalEntropy/runCount));
        
        
        datasetName++;
    }
        }
        catch(Exception e){
               System.out.println("Exception");
               System.out.println(e.getMessage()); 
          }
    }
    
    
    
}
