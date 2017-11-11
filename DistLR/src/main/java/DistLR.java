import org.kohsuke.args4j.Option;
import org.petuum.jbosen.PsApplication;
import org.petuum.jbosen.PsConfig;
import org.petuum.jbosen.PsTableGroup;

public class DistLR {
  
  private static class CmdArgs extends PsConfig {
    @Option(name = "-dataFile", required = true, usage = "Path to data file.")
    public String dataFile = "";
    
    @Option(name = "-outputDir", required = true, usage = "Path to output dir.")
    public String outputDir = "";
   
/* 
    @Option(name = "-numWords", required = true,
            usage = "Number of words in the vocabulary.")
    public int numWords = 0;
*/

/*
    @Option(name = "-numTopics", required = true, usage = "Number of topics.")
    public int numTopics = 0;
*/

    @Option(name = "-eta", required = true, usage = "Eta.")
    public double eta = 0.0;

    @Option(name = "-lambda_", required = true, usage = "Lambda.")
    public double lambda_ = 0.0;
    
    @Option(name = "-numIterations", required = true,
            usage = "Number of iterations.")
    public int numIterations = 0;
    
    @Option(name = "-numClocksPerIteration", required = true,
            usage = "Number of clocks for each iteration.")
    public int numClocksPerIteration = 0;
    
    @Option(name = "-staleness", required = false,
            usage = "Staleness of word-topic table. Default = 0")
    public int staleness = 0;
  }
  
  public static void main(String[] args) {
    CmdArgs cmd = new CmdArgs();
    cmd.parse(args);
    DistLRApp distLRApp = new DistLRApp(cmd.dataFile, cmd.outputDir, /*cmd.numWords,*/
                               /*cmd.numTopics,*/cmd.eta, cmd.lambda_,
                               cmd.numIterations, cmd.numClocksPerIteration,
                               cmd.staleness);
    distLRApp.run(cmd);
  }

}
