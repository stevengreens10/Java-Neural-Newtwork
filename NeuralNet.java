import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class NeuralNet {
	
	public int[] sizes;
	// [region] [node] [weight to node in previous layer]
	public double[][][] weights;
	// [layer] [node]
	public double[][] biases;

	public NeuralNet(int[] sizes) {
		this.sizes = sizes;
		
		biases = new double[sizes.length-1][1];
		weights = new double[sizes.length-1][1][1];
		
		// First layer has no biases; start loop at 1
		for(int i = 1; i < sizes.length; i++) {
			biases[i-1] = new double[sizes[i]];
			
			for(int j = 0; j < biases[i-1].length; j++) {
				biases[i-1][j] = new Random().nextGaussian();
			}
		}
		
		Random random = new Random();
		
		for(int i = 0; i < sizes.length-1; i++) {
			weights[i] = new double[ sizes[i+1] ][ sizes[i] ];
			
			for(int j = 0; j < weights[i].length; j++) {
				for(int k = 0; k < weights[i][j].length; k++) {
					weights[i][j][k] = random.nextGaussian();
//					System.out.println(weights[i][j][k]);
				}
			}
		}
		
	}
	
	public double[] feedforward(double[] a) {
		// loops through layers
		for(int i = 0; i < this.weights.length; i++) {
			// loops through nodes
			double[] temp = new double[this.sizes[i+1]];
			for(int j = 0; j < this.weights[i].length; j++) {
				double bias = this.biases[i][j];
				
				double sum = 0;
				// loops through weights
				// sums the product of each input and weight
				for(int k = 0; k < this.weights[i][j].length; k++) {
					sum += (this.weights[i][j][k] * a[k]);
				}
				
				
				// stores output for each node
				temp[j] = sigmoid(sum+bias);
			}
			
			// sets a equal to the output of previous layer 
			a = Arrays.copyOf(temp, temp.length);
		}
		return a;
	}
	
	public void evaluate(NetData data) {
		
		int correct = 0;
		int count = 0;
		
		for(double[] input : data.data.keySet()) {
			int actual = data.data.get(input);
			
			double[] activations = feedforward(input);
			double max = Integer.MIN_VALUE;
			int guess = 0;
//			System.out.println(Arrays.toString(activations));
			for(int i = 0; i < activations.length; i++) {
				if(activations[i] > max) {
					max = activations[i];
					guess = i;
				}
			}
			
			if(actual == guess) {
				correct++;
			}
			
//			System.out.printf("Guess: %d, Actual: %d | %.2f%% certainty\n", guess, actual, max*100);
			
			count++;
		}
		
		System.out.printf("%d / %d correct (%.2f%%)\n", correct, count, ((double) correct/count)*100);
	}
	
	/**
	 * Gradient Descent 
	 */
	public void learn(NetData trainData, NetData testData, int numEpochs, int batchSize, double learnRate) {
		for(int i = 0; i < numEpochs; i++) {
			
			List<double[]> keys = new ArrayList<double[]>(trainData.data.keySet());
			Collections.shuffle(keys);
			
			List<List<double[]>> batches = chunk(keys, batchSize);
			
			for(List<double[]> batchInput : batches) {
				List<Integer> outputs = new ArrayList<Integer>();
				for(double[] input : batchInput) {
					outputs.add(trainData.data.get(input));
				}
				NetData batch = new NetData(batchInput, outputs);
				updateMiniBatch(batch, learnRate);
			}
			
			System.out.println("Epoch " + i + " complete.");
			evaluate(testData);
		}
	}
	
	private void updateMiniBatch(NetData batch, double learnRate) {
		// Set up arrays with proper dimensions
		double[][] nablaBias = new double[this.biases.length][1];
		for(int i = 0; i < nablaBias.length; i++) {
			nablaBias[i] = new double[this.biases[i].length];
		}
		
		double[][][] nablaWeight = new double[this.weights.length][1][1];
		for(int i = 0; i < nablaWeight.length; i++) {
			nablaWeight[i] = new double[this.weights[i].length][1];
			for(int j = 0; j < nablaWeight[i].length; j++) {
				nablaWeight[i][j] = new double[this.weights[i][j].length];
			}
		}

		for(double[] input : batch.data.keySet()) {
			int realVal = batch.data.get(input);
			
			BackPropResult result = backprop(input, realVal);
			double[][][] dw = result.dw;
			double[][] db = result.db;
			
			for(int i = 0; i < nablaWeight.length; i++) {
				for(int j = 0; j < nablaWeight[i].length; j++) {
					
					nablaBias[i][j] += db[i][j]*(learnRate/batch.data.size());
					
					for(int k = 0; k < nablaWeight[i][j].length; k++) {
						nablaWeight[i][j][k] += dw[i][j][k]*(learnRate/batch.data.size());
					}
				}
			}
		}
		
		//Update weights and biases
		
		for(int i = 0; i < this.weights.length; i++) {
			for(int j = 0; j < this.weights[i].length; j++) {
				this.biases[i][j] += nablaBias[i][j];
				for(int k = 0; k < this.weights[i][j].length; k++) {
					this.weights[i][j][k] += nablaWeight[i][j][k];
				}
			}
		}
		
	}
	
	private BackPropResult backprop(double[] input, int realVal) {
		
		
		// turn output number into array of activations for output layer
		// 1.0 for expected number, 0.0 for anything else
		double[] realWeights = new double[this.sizes[this.sizes.length-1]];
		
		for(int i = 0; i < realWeights.length; i++) {
			realWeights[i] = (realVal == i) ? 1.0 : 0.0;
		}
		
		double[][] activations = new double[this.sizes.length][1];
		
		for(int i = 0; i < activations.length; i++) {
			activations[i] = new double[this.sizes[i]];
		}
		
		// gives correct dimensions
		double[][] zValues = new double[this.biases.length][1];
		for(int i = 0; i < zValues.length; i++) {
			zValues[i] = new double[this.biases[i].length];
		}
		
		for(int i = 0; i < activations[0].length; i++) {
			activations[0][i] = input[i];
		}
		
		for(int i = 0; i < this.weights.length; i++) {
			for(int j = 0; j < this.weights[i].length; j++) {
				double bias = this.biases[i][j];
				double sum = 0;
				for(int k = 0; k < this.weights[i][j].length; k++) {
					sum += this.weights[i][j][k] * activations[i][k];
				}
				zValues[i][j] = sum + bias;
				activations[i+1][j] = sigmoid(zValues[i][j]);
			}
		}
		
		// now that zValues and activation arrays are set up, on to formula
		// error array with length of output layer
		double[] delta = new double[this.sizes[this.sizes.length-1]];
		
		for(int i = 0; i < delta.length; i++) {
			delta[i] = ( activations[activations.length-1][i] - realWeights[i] ) * sigmoidPrime(zValues[zValues.length-1][i]);
		}
		
		
		double[][] nablaBias = new double[this.biases.length][1];
		for(int i = 0; i < nablaBias.length; i++) {
			nablaBias[i] = new double[this.biases[i].length];
		}
		
		double[][][] nablaWeight = new double[this.weights.length][1][1];
		for(int i = 0; i < nablaWeight.length; i++) {
			nablaWeight[i] = new double[this.weights[i].length][1];
			for(int j = 0; j < nablaWeight[i].length; j++) {
				nablaWeight[i][j] = new double[this.weights[i][j].length];
			}
		}
		
		// Nabla bias and nabla weight for output layer
		nablaBias[nablaBias.length-1] = delta.clone();
		for(int j = 0; j < nablaWeight[nablaWeight.length-1].length; j++) {
			for(int k = 0; k < nablaWeight[nablaWeight.length-1][j].length; k++) {
				nablaWeight[nablaWeight.length-1][j][k] = delta[j]*activations[activations.length-2][k];
			}
		}
		
		for(int i = 2; i < this.sizes.length; i++) {
			
			double[] temp = new double[this.sizes[i-1]];
			
			for(int j = 0; j < delta.length; j++) {
				double sp = sigmoidPrime(zValues[zValues.length-i][j]);
				// temp[n] = SUM(delta[m] * weight[m]) * sp
				for(int k = 0; k < temp.length; k++) {
					double sum = 0;
					
					for(int l = 0; l < delta.length; l++) {
//						System.out.println("weight: " + this.weights[this.weights.length-i+1][l][k]);
//						System.out.println("delta: " + delta[l]);
						sum += this.weights[this.weights.length-i+1][l][k]*delta[l];
					}
					
					temp[k] = sum * sp;
//					System.out.println(temp[k]);
				}
			}
			
			delta = temp.clone();
			
			nablaBias[nablaBias.length-i] = delta.clone();
			
			for(int j = 0; j < nablaWeight[nablaWeight.length-i].length; j++) {
				for(int k = 0; k < nablaWeight[nablaWeight.length-i][j].length; k++) {
					nablaWeight[nablaWeight.length-i][j][k] = delta[j]*activations[activations.length-i-1][k];
				}
			}
		}
		
		for(int i = 0; i < nablaBias.length; i++) {
			for(int j = 0; j < nablaBias[i].length; j++) {
//				System.out.println(i + " " + j + " : " + nablaBias[i][j]);
			}
		}
		
		return new BackPropResult(nablaWeight, nablaBias); 
	}

	private static <T> List<List<T>> chunk(List<T> input, int chunkSize) {

        int inputSize = input.size();
        int chunkCount = (int) Math.ceil(inputSize / (double) chunkSize);

        Map<Integer, List<T>> map = new HashMap<>(chunkCount);
        List<List<T>> chunks = new ArrayList<>(chunkCount);

        for (int i = 0; i < inputSize; i++) {

            map.computeIfAbsent(i / chunkSize, (ignore) -> {

                List<T> chunk = new ArrayList<>();
                chunks.add(chunk);
                return chunk;

            }).add(input.get(i));
        }

        return chunks;
    }
	
	private double sigmoid(double x) {
		return (1.0 / (1.0 + Math.exp(x)));
	}
	
	private double sigmoidPrime(double x) {
		return sigmoid(x) * (1 - sigmoid(x));
	}
	
	private class BackPropResult {
		
		public double[][][] dw;
		public double[][] db;
		
		public BackPropResult(double[][][] dw, double[][] db) {
			this.dw = dw;
			this.db = db;
		}
	}
	
}