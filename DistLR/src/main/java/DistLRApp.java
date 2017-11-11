import org.petuum.jbosen.PsApplication;
import org.petuum.jbosen.PsTableGroup;
import org.petuum.jbosen.row.double_.DenseDoubleRowUpdate;
import org.petuum.jbosen.row.double_.DoubleRow;
import org.petuum.jbosen.row.double_.DoubleRowUpdate;
import org.petuum.jbosen.row.int_.IntRow;
import org.petuum.jbosen.table.IntTable;
import org.petuum.jbosen.table.DoubleTable;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.special.*;

public class DistLRApp extends PsApplication {

  //private static final int TOPIC_TABLE = 0;
  //private static final int WORD_TOPIC_TABLE = 1;
	private static final int WVECTORS_TABLE = 0;
	private static final int LOSS_TABLE =1;
	private static double overfl = 20;
	static String[] targetLabels = { "American_film_directors", "Articles_containing_video_clips", "English-language_journals", "Windows_games", "American_people_of_Irish_descent", "Deaths_from_myocardial_infarction", "Guggenheim_Fellows", "Columbia_University_alumni", "Fellows_of_the_Royal_Society", "Major_League_Baseball_pitchers", "Harvard_University_alumni", "American_male_film_actors", "English-language_television_programming", "American_film_actresses", "American_male_television_actors", "American_films", "English-language_films", "Black-and-white_films", "American_drama_films", "Yale_University_alumni", "English-language_albums", "American_television_actresses", "American_comedy_films", "The_Football_League_players", "English_footballers", "British_films", "American_military_personnel_of_World_War_II", "Association_football_goalkeepers", "Serie_A_players", "Italian_footballers", "Association_football_midfielders", "Association_football_forwards", "English_cricketers", "Scottish_footballers", "French_films", "Insects_of_Europe", "Italian_films", "German_footballers", "Indian_films", "Main_Belt_asteroids", "Asteroids_named_for_people", "Rivers_of_Romania", "Russian_footballers", "Villages_in_the_Czech_Republic", "Association_football_defenders", "Australian_rules_footballers_from_Victoria_(Australia)", "Hindi-language_films", "Brazilian_footballers", "Villages_in_Turkey" };	
	
  private String outputDir;
/*
  private int numWords;
  private int numTopics;
*/  
  private double eta;
  private double lambda_;
  private int numIterations;
  private int numClocksPerIteration;
  private int staleness;
  private DataLoader dataLoader;
  private Random random;
  private int totLabels = 49;
  private int featureSize = 1001;

  public DistLRApp(String dataFile, String outputDir, /*int numWords, int numTopics,*/
                double eta, double lambda_, int numIterations,
                int numClocksPerIteration, int staleness) {
    this.outputDir = outputDir;
/*
    this.numWords = numWords;
    this.numTopics = numTopics;
*/  
    this.eta = eta;
    this.lambda_ = lambda_;
    this.numIterations = numIterations;
    this.numClocksPerIteration = numClocksPerIteration;
    this.staleness = staleness;
    this.dataLoader = new DataLoader(dataFile);
    this.random = new Random();
  }

  /*
  public double logDirichlet(double[] alpha) {
		double sumLogGamma=0.0;
		double logSumGamma=0.0;
		for (double value : alpha){
			sumLogGamma += Gamma.logGamma(value);
			logSumGamma += value;
		}
		return sumLogGamma - Gamma.logGamma(logSumGamma);
	}
*/
  
  /*
	public double logDirichlet(double alpha, int k) {
		return k * Gamma.logGamma(alpha) - Gamma.logGamma(k*alpha);
	}
	*/
  
  /*
	public double[] getRows(IntTable matrix, int columnId) {
		double[] rows = new double[this.numWords];
		for (int i = 0; i < this.numWords; i ++){
			rows[i] = (double) matrix.get(i, columnId);
		}
		return rows;
	}
	*/
  
	/*
  public double[] getColumns(int[][] matrix, int rowId){
		double[] cols = new double[this.numTopics];
		for (int i = 0; i < this.numTopics; i ++){
			cols[i] = (double) matrix[rowId][i];
		}
		return cols;
	}
	*/

  /*
	public double getLogLikelihood(IntTable wordTopicTable,
                                 int[][] docTopicTable) {
	  double lik = 0.0;
	  for (int k = 0; k < this.numTopics; k ++) {
		  double[] temp = this.getRows(wordTopicTable, k);
		  for (int w = 0; w < this.numWords; w ++) {
				 temp[w] += this.alpha;
		  }
		  lik += this.logDirichlet(temp);
		  lik -= this.logDirichlet(this.beta, this.numWords);
	  }
	  for (int d = 0; d < docTopicTable.length; d ++) {
		  double[] temp = this.getColumns(docTopicTable, d);
		  for (int k = 0; k < this.numTopics; k ++) {
			 temp[k] += this.alpha;
		  }
		  lik += this.logDirichlet(temp);
		  lik -= this.logDirichlet(this.alpha, this.numTopics);
	  }
	  return lik;
  }
  */
  
  @Override
  public void initialize() {
    // Create global topic count table. This table only has one row, which
    // contains counts for all topics.
    
	 //PsTableGroup.createDenseIntTable(TOPIC_TABLE, staleness, numTopics);
    
	  // Create global word-topic table. This table contains numWords rows, each
    // of which has numTopics columns.
    
	  PsTableGroup.createDenseDoubleTable(WVECTORS_TABLE, staleness, totLabels);
	  PsTableGroup.createDenseDoubleTable(LOSS_TABLE, staleness,1);
  }

  @Override
  public void runWorkerThread(int threadId) {
    int clientId = PsTableGroup.getClientId();

    // Load data for this thread.
    System.out.println("Client " + clientId + " thread " + threadId +
                       " loading data...");
    int part = PsTableGroup.getNumLocalWorkerThreads() * clientId + threadId;
    int numParts = PsTableGroup.getNumTotalWorkerThreads();
    ArrayList<String> data = this.dataLoader.load(part, numParts);

    // Get global tables.
    DoubleTable wVectors = PsTableGroup.getDoubleTable(WVECTORS_TABLE);
    DoubleTable lossTable = PsTableGroup.getDoubleTable(LOSS_TABLE);
   // IntTable Aks = PsTableGroup.getIntTable(AKS);
    
    //IntTable wordTopicTable = PsTableGroup.getIntTable(WORD_TOPIC_TABLE);
    
    // Initialize Distributed Logistic Regression variables.
    System.out.println("Client " + clientId + " thread " + threadId +
                       " initializing variables...");
    
    //int[][] docTopicTable = new int[w.length][this.numTopics];
    
    //
    // ... fill me out ...
    
    
    
    for(int j=0;j<featureSize;j++)
    {
    	DoubleRowUpdate RUpdates = new DenseDoubleRowUpdate(totLabels);
    for(int i=0;i<totLabels;i++)
    {
    	RUpdates.set(i, 0);	
    }
    wVectors.inc(j, RUpdates);
    }
    
    
    for(int i=0;i<this.numIterations;i++)
    {
    	DoubleRowUpdate LUpdates = new DenseDoubleRowUpdate(1);
    	LUpdates.set(0, 0);	
    	lossTable.inc(i, LUpdates);
    }
    
   
    //

    // Global barrier to synchronize word-topic table.
    PsTableGroup.globalBarrier();

    // Do LDA Gibbs sampling.
    System.out.println("Client " + clientId + " thread " + threadId +
                       " starting Distributed LR...");
    double[] llh = new double[this.numIterations];
    double[] sec = new double[this.numIterations];
    double totalSec = 0.0;
    double l = 0;
    int counter = 0;
    
  //  for (int iter = 0; iter < this.numIterations; iter ++) {
     
      // Each iteration consists of a number of batches, and we clock
      // between each to communicate parameters according to SSP.
      for (int batch = 1; batch <= this.numIterations; batch ++) {
    	  long startTime = System.currentTimeMillis();
    	  DoubleRowUpdate linc = new DenseDoubleRowUpdate(1);
       // int begin = w.length * batch / this.numClocksPerIteration;
       // int end = w.length * (batch + 1) / this.numClocksPerIteration;
        // Loop through each document in the current batch.
        //for (int d = begin; d < end; d ++) {
          //
          // ... fill me out ...
    	  
    	    l = 0;
    	    counter = 0;
			System.out.println(batch+","+this.numIterations);
			eta = eta / (batch);
			
			for (String document : data)
		    {
				
				counter += 1;
				String label = document.split("\\s", 2)[0];
				int[] labelVector = getLabelsVector(label, targetLabels);
				Map<Integer, Integer> hashedFeatures = getHashedFeatures(document.split("\\s", 2)[1]);
				///////////////////////////////////// GRADIENT UPDATE ///////////////////////////////////
				gradientUpdate(labelVector, hashedFeatures, counter, wVectors);
				
				////////////////////////////////////////////////////////////////////////////////////////
				PsTableGroup.clock();
			}
			 long endTime = System.currentTimeMillis();
			 System.out.println("Epoch time ="+(endTime-startTime));
			 PsTableGroup.globalBarrier();
			
			if (clientId == 0 && threadId == 0) {
				l = calcLoss(data,wVectors);
			        l = l*(1/counter) + lambda_*calcNormsW(wVectors);
			        linc.set(0,l);
					lossTable.inc(batch-1,linc);
			          
			}
			
			
			
			//lazyUpdate(counter);
			//br.close();
		}
          //
        
        // Call clock() to indicate an SSP boundary.
        
      //}
      // Calculate likelihood and elapsed time.
      //totalSec += (double) (System.currentTimeMillis() - startTime) / 1000; 
      //sec[iter] = totalSec;
    //  llh[iter] = this.getLogLikelihood(wordTopicTable, docTopicTable);
     // System.out.println("Client " + clientId + " thread " + threadId +
          //               " completed iteration " + (iter + 1) +
            //             "\n    Elapsed seconds: " + sec[iter] +
              //           "\n    Log-likelihood: " + llh[iter]);
   // }

    PsTableGroup.globalBarrier();

   
//    PsTableGroup.globalBarrier();
    
    // Output tables.
    if (clientId == 0 && threadId == 0) {
      System.out.println("Client " + clientId + " thread " + threadId +
                         " writing wVectors table to file...");
      try {
        PrintWriter out = new PrintWriter(this.outputDir + "/wVectors.csv");
      
        
        for (int i = 0; i < totLabels; i ++) {
          out.print(wVectors.get(i, 0));
          for (int k = 0; k < featureSize; k ++) {
            out.print("," + wVectors.get(k, i));
          }
          out.println();
        }
      
        out.close();
      } catch (IOException e) {
        e.printStackTrace();
        System.exit(1);
      }
      
      System.out.println("Client " + clientId + " thread " + threadId +
              " writing loss table to file...");
      try {
    	  	PrintWriter out = new PrintWriter(this.outputDir + "/loss.csv");
    	  	for (int i = 0; i < this.numIterations; i ++) {
    	  		out.println(","+lossTable.get(i,0));
    	  	}
    	  		out.println();
    	  	
    	  	out.close();
      } catch (IOException e) {
    	  	e.printStackTrace();
    	  	System.exit(1);
      	}
      System.out.println("Client " + clientId + " thread " + threadId +
              " writing loss table to file...");
      try {
    	  long startTime = System.currentTimeMillis();
    	  testLR("testsmallnew.csv", wVectors);
    	  long endTime = System.currentTimeMillis();
    	  testLR("trainsmallnew.csv", wVectors);
    	  long endTime2 = System.currentTimeMillis();
    	  System.out.println("Test test time = "+(endTime-startTime));
    	  System.out.println("Test test time = "+(endTime2-endTime));
    	  	
      } catch (IOException e) {
    	  	e.printStackTrace();
    	  	System.exit(1);
      	}
     
    }

    //PsTableGroup.globalBarrier();

    System.out.println("Client " + clientId + " thread " + threadId + " exited.");
  }


  private double calcLoss(ArrayList<String> data, DoubleTable wVectors)
	{
		double l=0;
		int lab;
		//int count = 0;
		double prob = 0;
		double normsW = 0;
		for(String document : data) {
			String label = document.split("\\s", 2)[0];
			int[] labelVector = getLabelsVector(label, targetLabels);
			Map<Integer, Integer> hashedFeatures = getHashedFeatures(document.split("\\s", 2)[1]);
			lab = java.util.Arrays.asList(targetLabels).indexOf(label);
			//System.out.println(label+" "+lab);
			if(lab>=0)
			{
				prob = docPredLabel(hashedFeatures, lab, wVectors);
				l = l + Math.log(prob);
			//	count = count + 1;
			}			
		}
		
		//normsW = calcNormsW(wVectors);
		l = -l;// + lambda_*normsW;
		//System.out.println(prob+" "+l);
		return l;
	}

	private double calcNormsW(DoubleTable wVectors)
	{
		double w = 0;
		double norms = 0;
		for(int i=0;i<(featureSize-1);i++)
		{
			for(int j=0;j<totLabels;j++)
			 w = wVectors.get(i, j);
			 norms = norms + w*w;
			 
		}
		return norms;
	}
	
	private double docPredLabel(Map<Integer, Integer> hashedFeatures, int label, DoubleTable wVectors) {
		int nLabels = targetLabels.length;
		DoubleRow bias = wVectors.get(featureSize-1);
		double [] prob = new double[targetLabels.length];
		double normW = 0;
		double sumprob = 0;
		for(int i=0;i<targetLabels.length;i++)
		{
			prob[i] = bias.get(i);
		}
		
		for(int i=0;i<targetLabels.length;i++)
		{
		for (Integer hashVal : hashedFeatures.keySet()) {
			int count = hashedFeatures.get(hashVal);
			
			if (wVectors.get(hashVal,label)>0.0) {
				DoubleRow w = wVectors.get(hashVal);
				prob[i] += w.get(i)*count;
				
			}
		}
		}
		for(int i=0;i<targetLabels.length;i++)
		{
			prob[i] = Math.exp(prob[i]);
			sumprob += prob[i];
		}
		return(prob[label]/sumprob);
	}
  
private Map<Integer, Integer> getHashedFeatures(String curDoc) {
	
	String[] keyValPair = curDoc.split("\\s+");
	Map<Integer, Integer> hashedFeatures = new HashMap<>();
	
	for (int i = 0; i < keyValPair.length; i++) {
			String[] str = keyValPair[i].split(",");
			//System.out.println(keyValPair[i]);
			if(str[0].length()>0)
			{
			   hashedFeatures.put(Integer.parseInt(str[0]), (int)Float.parseFloat(str[1]));
			}
		
	}
	return hashedFeatures;
}

private int[] getLabelsVector(String labelString, String[] targetLabels) {
	int[] labels = new int[targetLabels.length];
	for (int i = 0; i < targetLabels.length; i++) {
		labels[i] = labelString.contains(targetLabels[i]) ? 1 : 0;
	}
	return labels;
}

private double logistic(double score) {
	if (Math.abs(score) > overfl) {
		score = Math.signum(score) * overfl;
	}
	return 1 / (1 + Math.exp(-score));
}

private double[] docPred(Map<Integer, Integer> hashedFeatures, DoubleTable wVectors) {
	int nLabels = targetLabels.length;
	DoubleRow bias = wVectors.get(featureSize-1);
	double[] prob = new double[nLabels];
	
	for (int i = 0; i < nLabels; i++) {
		prob[i] = bias.get(i);
	}
	
	for (Integer hashVal : hashedFeatures.keySet()) {
		int count = hashedFeatures.get(hashVal);
		
		//if (wVectors.get(hashVal)) {
			DoubleRow w = wVectors.get(hashVal);
			for (int i = 0; i < nLabels; i++) {
				prob[i] += w.get(i) * count;
			}
		//}
	}

	for (int i = 0; i < nLabels; i++) {
		prob[i] = logistic(prob[i]);
	}
	return prob;
}

private void gradientUpdate(int[] labelVector, Map<Integer, Integer> hashedFeatures, int counter, DoubleTable wVectors) 
{
	int nLabels = targetLabels.length;
	double val=0;
	// predicted probabilities using current model parameters
	double[] predProb = docPred(hashedFeatures, wVectors);
	
	// update bias term
	DoubleRowUpdate bias = new DenseDoubleRowUpdate(totLabels);
	for (int i = 0; i < nLabels; i++) {
		
		val = eta * (labelVector[i] - predProb[i]);
		bias.set(i,val); 
	}
	wVectors.inc(featureSize-1, bias);
	 
	int count=0;
	 for(int i=0;i<featureSize;i++) {
		 if(hashedFeatures.containsKey(i))
		 {
		count = hashedFeatures.get(i);
		 }
		 else
		 {
			 count = 0;
		 }
		//DoubleRow w;
		DoubleRowUpdate wtemp = new DenseDoubleRowUpdate(totLabels);
		//IntRow lazyUpdateParams;
		
		
		for (int j = 0; j < nLabels; j++) {
			wtemp.set(j, eta * (labelVector[j] - predProb[j])*count -2*eta*lambda_*wVectors.get(i,j));			
		}
		
		wVectors.inc(i, wtemp);
		//Aks.put(hashVal, lazyUpdateParams);
	}
}

public void testLR(String test,DoubleTable wVectors) throws IOException {
	
	int accuracy = 0, totalDoc = 0;
	
	BufferedReader br = new BufferedReader(new FileReader(test));
	String testDoc;
	while ((testDoc = br.readLine()) != null) {
		String label = testDoc.split("\\s", 2)[0];
		Map<Integer, Integer> hashedFeatures = getHashedFeatures(testDoc.split("\\s", 2)[1]);
		double[] predictions = docPred(hashedFeatures,wVectors);
		int[] labels = getLabelsVector(label,targetLabels);
		int argmax = 0; 
		double maxVal = 0;
		for (int i = 0; i < targetLabels.length; i++) {
			if ( predictions[i] > maxVal) {
				maxVal = predictions[i];
				argmax = i;
			}
		}
		if (labels[argmax] == 1) {
			accuracy += 1;
		}
		totalDoc = totalDoc + 1;
	}
	System.out.println(accuracy);
	System.out.println(totalDoc);
	System.out.println((double)accuracy / totalDoc);
	br.close();
}

}

