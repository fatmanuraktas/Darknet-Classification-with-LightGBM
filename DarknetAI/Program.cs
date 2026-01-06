using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace DarknetAI
{
    // --- CSV COLUMNS ---
    public class ModelInput
    {
    [LoadColumn(0)] public float Protocol { get; set; }
    [LoadColumn(1)] public float FlowDuration { get; set; }
    [LoadColumn(2)] public float TotFwdPkts { get; set; }
    [LoadColumn(3)] public float TotBwdPkts { get; set; }
    [LoadColumn(4)] public float TotLenFwdPkts { get; set; }
    [LoadColumn(5)] public float TotLenBwdPkts { get; set; }
    [LoadColumn(6)] public float FwdPktLenMax { get; set; }
    [LoadColumn(7)] public float FwdPktLenMin { get; set; }
    [LoadColumn(8)] public float FwdPktLenMean { get; set; }
    [LoadColumn(9)] public float FwdPktLenStd { get; set; }
    [LoadColumn(10)] public float BwdPktLenMax { get; set; }
    [LoadColumn(11)] public float BwdPktLenMin { get; set; }
    [LoadColumn(12)] public float BwdPktLenMean { get; set; }
    [LoadColumn(13)] public float BwdPktLenStd { get; set; }
    [LoadColumn(14)] public float FlowBytss { get; set; }
    [LoadColumn(15)] public float FlowPktss { get; set; }
    [LoadColumn(16)] public float FlowIATMean { get; set; }
    [LoadColumn(17)] public float FlowIATStd { get; set; }
    [LoadColumn(18)] public float FlowIATMax { get; set; }
    [LoadColumn(19)] public float FlowIATMin { get; set; }
    [LoadColumn(20)] public float FwdIATMean { get; set; }
    [LoadColumn(21)] public float FwdIATStd { get; set; }
    [LoadColumn(22)] public float FwdIATMax { get; set; }
    [LoadColumn(23)] public float FwdIATMin { get; set; }
    [LoadColumn(24)] public float BwdIATMean { get; set; }
    [LoadColumn(25)] public float BwdIATStd { get; set; }
    [LoadColumn(26)] public float BwdIATMax { get; set; }
    [LoadColumn(27)] public float BwdIATMin { get; set; }
    [LoadColumn(28)] public float FwdHeaderLen { get; set; }
    [LoadColumn(29)] public float BwdHeaderLen { get; set; }
    [LoadColumn(30)] public float FwdPktss { get; set; }
    [LoadColumn(31)] public float BwdPktss { get; set; }
    [LoadColumn(32)] public float PktLenMin { get; set; }
    [LoadColumn(33)] public float PktLenMax { get; set; }
    [LoadColumn(34)] public float PktLenMean { get; set; }
    [LoadColumn(35)] public float PktLenStd { get; set; }
    [LoadColumn(36)] public float PktLenVar { get; set; }
    [LoadColumn(37)] public float FwdSegSizeAvg { get; set; }
    [LoadColumn(38)] public float InitFwdWinByts { get; set; }
    [LoadColumn(39)] public float InitBwdWinByts { get; set; }
    [LoadColumn(40)] public float FwdActDataPkts { get; set; }
    [LoadColumn(41)] public float ActiveMean { get; set; }
    [LoadColumn(42)] public float ActiveStd { get; set; }
    [LoadColumn(43)] public float ActiveMax { get; set; }
    [LoadColumn(44)] public float ActiveMin { get; set; }
    [LoadColumn(45)] public float IdleMean { get; set; }
    [LoadColumn(46)] public float IdleStd { get; set; }
    [LoadColumn(47)] public float IdleMax { get; set; }
    [LoadColumn(48)] public float IdleMin { get; set; }
    [LoadColumn(49), ColumnName("Label")] public float Label { get; set; }
    }

    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Score { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);
            
            string dataPath = "csharp_final_ready.csv"; 

            Console.WriteLine("1. Data loading...");
            
            try 
            {
                if (!System.IO.File.Exists(dataPath))
                {
                    Console.WriteLine($"ERROR: '{dataPath}' file not found.");
                    return;
                }

                IDataView dataView = context.Data.LoadFromTextFile<ModelInput>(
                    path: dataPath,
                    hasHeader: true, 
                    separatorChar: ',');

                var split = context.Data.TrainTestSplit(dataView, testFraction: 0.2);

                Console.WriteLine("2nd Model preparing...");

                var featureColumnNames = typeof(ModelInput).GetProperties()
                    .Where(p => p.Name != "Label") 
                    .Select(p => p.Name)
                    .ToArray();

                var pipeline = context.Transforms.Concatenate("Features", featureColumnNames)
                    .Append(context.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Boolean))
                    .Append(context.BinaryClassification.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "Features", numberOfIterations: 100));

                Console.WriteLine("3rd Training...");
                var model = pipeline.Fit(split.TrainSet);

                Console.WriteLine("4th Testing...");
                var predictions = model.Transform(split.TestSet);
                var metrics = context.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");

                Console.WriteLine($"\n--- RESULTS ---");
                Console.WriteLine($"Accuracy : {metrics.Accuracy:P2}");
                Console.WriteLine($"F1 Score : {metrics.F1Score:P2}");
                Console.WriteLine($"----------------");
                
                context.Model.Save(model, dataView.Schema, "DarknetModel.zip");
                Console.WriteLine("Model saved succesfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CRITICAL ERROR: {ex.Message}");
            }
        }
    }
}