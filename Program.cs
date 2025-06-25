using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace AIProject
{
    public class StudentData
    {
        [LoadColumn(0)]
        public float GodzinyNauki { get; set; }

        [LoadColumn(1)]
        public float Frekwencja { get; set; }

        [LoadColumn(2)]
        [ColumnName("Label")]
        public bool ZdalEgzamin { get; set; }
    }

    public class StudentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedZdalEgzamin { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            IDataView data = mlContext.Data.LoadFromTextFile<StudentData>("C:\\Users\\paulinka\\Desktop\\c++\\4 semestr\\AIProject\\dane.csv", separatorChar: ',', hasHeader: true);

            // Używamy Normalization do skalowania wartości liczbowych do zakresu [0, 1]
            var dataProcessPipeline = mlContext.Transforms.NormalizeMinMax("GodzinyNauki")
    .Append(mlContext.Transforms.NormalizeMinMax("Frekwencja"))
    .Append(mlContext.Transforms.Concatenate("Features", "GodzinyNauki", "Frekwencja"))
    .AppendCacheCheckpoint(mlContext);

            var preprocessedData = dataProcessPipeline.Fit(data).Transform(data);

            // Podział danych na zbiór treningowy i testowy (80% trening, 20% test)
            // Używamy tego podziału dla ogólnego testowania, ale dla oceny modelu będziemy używać kroswalidacji.
            var trainTestData = mlContext.Data.TrainTestSplit(preprocessedData, testFraction: 0.2);
            IDataView trainingData = trainTestData.TrainSet;
            IDataView testData = trainTestData.TestSet;

            // Używamy FastTreeBinaryClassifier jako jeden z modeli, który dobrze radzi sobie z danymi tabelarycznymi
            Console.WriteLine("--- Ocena modelu FastTreeBinaryClassifier z 10-krotną kroswalidacją ---");
            var fastTreeModel = mlContext.BinaryClassification.Trainers.FastTree(numberOfLeaves: 4, numberOfTrees: 4, minimumExampleCountPerLeaf: 2);

            var crossValidationResultsFastTree = mlContext.BinaryClassification.CrossValidate(trainingData, fastTreeModel, numberOfFolds: 5);

            // Oblicz średnią dokładność
            var averageAccuracyFastTree = crossValidationResultsFastTree.Select(model => model.Metrics.Accuracy).Average();
            Console.WriteLine($"Średnia dokładność FastTree: {averageAccuracyFastTree:P2}");
            Console.WriteLine($"Średnia F1-Score FastTree: {crossValidationResultsFastTree.Select(model => model.Metrics.F1Score).Average():P2}");
            Console.WriteLine($"Średnia AUC FastTree: {crossValidationResultsFastTree.Select(model => model.Metrics.AreaUnderRocCurve).Average():P2}");

            // Tworzymy dwa dodatkowe modele do komitetu: Logistic Regression i LightGBM.
            Console.WriteLine("\n--- Trenowanie modeli dla komitetu ---");

            var logisticRegressionModel = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression();
            var lightGbmModel = mlContext.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
            {
                NumberOfLeaves = 8, // Maksymalna liczba liści w drzewie
                NumberOfIterations = 10, // Liczba drzew wzmacniających (boosted trees)
                MinimumExampleCountPerLeaf = 4, // Minimalna liczba przykładów w liściu

                LabelColumnName = "Label",
                FeatureColumnName = "Features"
            });

            var fastTreeTrainingPipeline = dataProcessPipeline.Append(fastTreeModel);
            var logisticRegressionTrainingPipeline = dataProcessPipeline.Append(logisticRegressionModel);
            var lightGbmTrainingPipeline = dataProcessPipeline.Append(lightGbmModel); 
            // Trenowanie modeli
            ITransformer trainedFastTreeModel = fastTreeTrainingPipeline.Fit(trainingData);
            ITransformer trainedLogisticRegressionModel = logisticRegressionTrainingPipeline.Fit(trainingData);
            ITransformer trainedLightGbmModel = lightGbmTrainingPipeline.Fit(trainingData);


            // Tworzenie silników predykcji
            var predEngineFastTree = mlContext.Model.CreatePredictionEngine<StudentData, StudentPrediction>(trainedFastTreeModel);
            var predEngineLogisticRegression = mlContext.Model.CreatePredictionEngine<StudentData, StudentPrediction>(trainedLogisticRegressionModel);
            var predEngineLightGbm = mlContext.Model.CreatePredictionEngine<StudentData, StudentPrediction>(trainedLightGbmModel);

            // Przeprowadzenie predykcji na zbiorze testowym i zbieranie wyników dla komitetu
            var predictionsFastTree = mlContext.Data.CreateEnumerable<StudentPrediction>(trainedFastTreeModel.Transform(testData), reuseRowObject: false).ToList();
            var predictionsLogisticRegression = mlContext.Data.CreateEnumerable<StudentPrediction>(trainedLogisticRegressionModel.Transform(testData), reuseRowObject: false).ToList();
            var predictionsLightGbm = mlContext.Data.CreateEnumerable<StudentPrediction>(trainedLightGbmModel.Transform(testData), reuseRowObject: false).ToList();

            Console.WriteLine("\n--- Wyniki głosowania większościowego (komitetu) ---");
            int correctCommitteePredictions = 0;
            int totalTestSamples = 0;

            // Pobieramy oryginalne etykiety ze zbioru testowego, aby porównać z predykcjami komitetu
            var originalTestLabels = mlContext.Data.CreateEnumerable<StudentData>(testData, reuseRowObject: false).Select(s => s.ZdalEgzamin).ToList();

            for (int i = 0; i < predictionsFastTree.Count; i++)
            {
                bool predictionFastTree = predictionsFastTree[i].PredictedZdalEgzamin;
                bool predictionLogisticRegression = predictionsLogisticRegression[i].PredictedZdalEgzamin;
                bool predictionLightGbm = predictionsLightGbm[i].PredictedZdalEgzamin;
                bool trueLabel = originalTestLabels[i];

                int yesVotes = 0;
                if (predictionFastTree) yesVotes++;
                if (predictionLogisticRegression) yesVotes++;
                if (predictionLightGbm) yesVotes++;

                bool committeePrediction = yesVotes >= 2; // Głosowanie większościowe

                Console.WriteLine($"Sample {i + 1}: FastTree: {predictionFastTree}, LogisticReg: {predictionLogisticRegression}, LightGBM: {predictionLightGbm}, Komitet: {committeePrediction}, Prawda: {trueLabel}");

                if (committeePrediction == trueLabel)
                {
                    correctCommitteePredictions++;
                }
                totalTestSamples++;
            }

            double committeeAccuracy = (double)correctCommitteePredictions / totalTestSamples;
            Console.WriteLine($"\nDokładność komitetu: {committeeAccuracy:P2}");


            Console.WriteLine("\n--- Symulacja predykcji dla nowych danych ---");
            var newStudent1 = new StudentData { GodzinyNauki = 6.5f, Frekwencja = 90f };
            var newStudent2 = new StudentData { GodzinyNauki = 2.5f, Frekwencja = 55f };

            // Używamy głównego modelu FastTree do predykcji na nowych danych
            var prediction1 = predEngineFastTree.Predict(newStudent1);
            var prediction2 = predEngineFastTree.Predict(newStudent2);

            Console.WriteLine($"Student 1 (GodzinyNauki: {newStudent1.GodzinyNauki}, Frekwencja: {newStudent1.Frekwencja}) => ZdalEgzamin: {prediction1.PredictedZdalEgzamin} (Prawdopodobieństwo: {prediction1.Probability:P2})");
            Console.WriteLine($"Student 2 (GodzinyNauki: {newStudent2.GodzinyNauki}, Frekwencja: {newStudent2.Frekwencja}) => ZdalEgzamin: {prediction2.PredictedZdalEgzamin} (Prawdopodobieństwo: {prediction2.Probability:P2})");

            Console.WriteLine("\n--- Dane do wykresów ---");
            Console.WriteLine("Do stworzenia wykresów możesz użyć poniższych danych:");
            Console.WriteLine($"Średnia dokładność FastTree (Cross-Validation): {averageAccuracyFastTree}");
            Console.WriteLine($"Dokładność komitetu (na zbiorze testowym): {committeeAccuracy}");
            Console.WriteLine("\nSzczegółowe wyniki kroswalidacji dla FastTree:");
            foreach (var foldResult in crossValidationResultsFastTree.Select((r, i) => new { Result = r, Index = i }))
            {
                Console.WriteLine($"  Fold {foldResult.Index + 1}: Accuracy = {foldResult.Result.Metrics.Accuracy:P2}, F1-Score = {foldResult.Result.Metrics.F1Score:P2}");
            }

            Console.ReadKey();
        }
    }
}