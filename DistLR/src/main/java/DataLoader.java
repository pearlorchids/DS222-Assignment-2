import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class DataLoader {

  private String dataFile;

  public DataLoader(String dataFile) {
    this.dataFile = dataFile;
  }

  public ArrayList<String> load(int part, int numParts) {
    ArrayList<String> lines = new ArrayList<String>();
    try {
      FileReader fr = new FileReader(this.dataFile);
      BufferedReader br = new BufferedReader(fr);
      int i = 0;
      while (true) {
        String line = br.readLine();
        if (line == null) {
          break;
        } else if (i % numParts == part) {
          lines.add(line);
        }
        i ++;
      }
      br.close();
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(1);
    }
    
    return lines;
  }

}
