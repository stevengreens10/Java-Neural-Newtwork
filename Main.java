public class Main {
	public static void main(String args[]) {
		NeuralNet nn = new NeuralNet(new int[] {784,30,10});
		
		MNISTLoader trainMNIST = new MNISTLoader(System.getProperty("user.home") + "/Documents/train-images.idx3-ubyte", 
				System.getProperty("user.home") + "/Documents/train-labels.idx1-ubyte", 20000);
		MNISTLoader testMNIST = new MNISTLoader(System.getProperty("user.home") + "/Documents/t10k-images.idx3-ubyte", 
				System.getProperty("user.home") + "/Documents/t10k-labels.idx1-ubyte", 10000);
		
		NetData trainData = trainMNIST.load();
		NetData testData = testMNIST.load();
		
		nn.learn(trainData, testData, 100, 10, 1);
	}
}
