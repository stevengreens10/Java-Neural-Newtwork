import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class MNISTLoader {

	private File imgFile;
	private File labelFile;
	private int numValues;
	
	public MNISTLoader(String imgFilePath, String labelFilePath, int maxNumValues) {
		imgFile = new File(imgFilePath);
		labelFile = new File(labelFilePath);
		numValues = maxNumValues;
	}
	
	public NetData load() {
		FileInputStream imgIn = null;
		FileInputStream labelIn = null;
		List<double[]> inputList = null;
		List<Integer> outputList = null;
		try {
			imgIn = new FileInputStream(imgFile);
			byte[] imgBytes = new byte[(int) imgFile.length()];
			imgIn.read(imgBytes);
			
			ByteBuffer bb = ByteBuffer.wrap(imgBytes);
			
			// Magic number
			bb.getInt();
			int numImages = bb.getInt();
			int numRows = bb.getInt();
			int numColumns = bb.getInt();
			
			inputList = new ArrayList<double[]>();
			
			if(numValues > numImages) {
				numValues = numImages;
			}
			
			for(int i = 0; i < numValues; i++) {
				double[] input = new double[numRows*numColumns];
				for(int j = 0; j < input.length; j++) {
					byte signed = bb.get();
					int unsignedByteVal = signed & 0xff;
					input[j] = map(unsignedByteVal, 0, 255, 0, 1);
				}
				inputList.add(input);
			}
			
			labelIn = new FileInputStream(labelFile);
			byte[] labelBytes = new byte[(int) labelFile.length()];
			labelIn.read(labelBytes);
			bb = ByteBuffer.wrap(labelBytes);
			// Magic number
			bb.getInt();
			// Number of labels
			bb.getInt();
			
			outputList = new ArrayList<Integer>();
			
			for(int i = 0; i < inputList.size(); i++) {
				byte signed = bb.get();
				int output = signed & 0xff;
				outputList.add(output);
			}
			
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return new NetData(inputList, outputList);
	}
	
	private double map(double value, double start1, double stop1, double start2, double stop2) {
		return ((value-start1)/(stop1-start1) * (stop2-start2) + start2);
	}
	
}
