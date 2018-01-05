import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NetData {
	public Map<double[], Integer> data = new HashMap<double[], Integer>();
	
	public NetData() {}
	
	public NetData(List<double[]> inputs, List<Integer> outputs) {
		if(inputs.size() == outputs.size()) {
			for(int i = 0; i < inputs.size(); i++) {
				data.put(inputs.get(i), outputs.get(i));
			}
		}
	}
}
