package digitRecognizer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

public class Util {
	/*
	 * Load ARFF file, if the file does not exist generate the ARFF file using
	 * method: csvToArff
	 */
	public static Instances loadData(String arffFileName, boolean isTrain)
			throws Exception {
		File arffFile = new File(arffFileName);
		if (!arffFile.exists() || arffFileName.endsWith(".csv")) {
			if (arffFileName.endsWith(".arff") || arffFileName.endsWith(".csv")) {
				System.err
						.println("ARFF file does not exist, converting CSV to ARFF");
				arffFileName = arffFileName.replace(".csv", ".arff");
				String csvFileName = arffFileName.replace(".arff", ".csv");
				if (!csvToArffTrain(csvFileName, isTrain)) {
					System.err.println("Fail: converting csv to arff");
					return null;
				}
			} else {
				return null;
			}
		}

		BufferedReader reader = new BufferedReader(new FileReader(arffFileName));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		if (isTrain)
			data.setClassIndex(0);
		reader.close();
		return data;
	}

	public static boolean csvToArffTrain(String csvFileName, boolean isTrain)
			throws Exception {
		File csvFile = new File(csvFileName);
		if (!csvFile.exists() || !csvFileName.endsWith(".csv")) {
			System.err.println("CSV file does not exist");
			return false;
		}

		String arffFileName = csvFileName.replace(".csv", ".arff");

		// load CSV
		CSVLoader loader = new CSVLoader();
		loader.setSource(csvFile);
		if (isTrain)
			loader.setNominalAttributes("first");
		Instances data = loader.getDataSet();

		// save ARFF
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File(arffFileName));
		saver.writeBatch();

		return true;
	}

	public static Instances addAttribute(Instances testData, String label) throws Exception {
		Instances newData = null;
		newData = new Instances(testData);
		Add filter;
		// 1. nominal attribute
		filter = new Add();
		filter.setAttributeIndex("first");
		filter.setNominalLabels(label);
		filter.setAttributeName("label");
		filter.setInputFormat(newData);
		newData = Filter.useFilter(newData, filter);
		
		return newData;
	}
}
