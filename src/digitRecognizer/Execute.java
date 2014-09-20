package digitRecognizer;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class Execute {

	public static void main(String[] args) throws Exception {
		
		System.out.println("Loading training data");
		Instances trainSet = Util.loadData(args[0], true);
		if (trainSet == null)
			throw new RuntimeException();
		String label = trainSet.attribute("label").toString();
		label = label.substring(label.indexOf("{")+1, label.indexOf("}"));
		System.out.println(label);
		System.out.println("Training data finished loading\n");

		System.out.println("Loading test data");
		Instances testSet = Util.loadData(args[1], false);
		if (testSet == null)
			throw new RuntimeException();
		testSet = Util.addAttribute(testSet, label);
		System.out.println("Test data finished loading\n");
		
		J48 tree = new J48();
		PrintStream ps = new PrintStream(new FileOutputStream(args[2]));

		System.out.println("Building classifier");
		trainSet.setClassIndex(0);
		testSet.setClassIndex(0);
		tree.buildClassifier(trainSet);
		
		System.out.println("# of test set: " + testSet.size());
		int i = 0;
		ps.println("ImageId,Label");
		while (i < testSet.size()) {
			double prediction = tree.classifyInstance(testSet.get(i));
			i++;
			ps.println(i + "," + trainSet.classAttribute().value((int)prediction));
		}
		ps.close();

	}
}
