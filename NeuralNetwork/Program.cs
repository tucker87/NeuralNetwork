using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworkNS.Main;

namespace NeuralNetworkNS.Console
{
    class Program
    {
        static void Main(string[] args)
        {
            var random = new Random();

            var nn = new NeuralNetwork(2, 2, 1);

            var result = nn.Predict(new double[] {0, 1});
            System.Console.WriteLine($"Before: {result}");

            var data = new List<(double[] inputs, double[] outputs)>
            {
                (new double[] { 0, 0 }, new double[]{0}),
                (new double[] { 1, 1 }, new double[]{0}),
                (new double[] { 0, 1 }, new double[]{1}),
                (new double[] { 1, 0 }, new double[]{1})
            };

            foreach (var _ in Enumerable.Range(0, 99))
            {
                var index = random.Next(0, 4);
                var trainingData = data[index];
                nn.Train(trainingData.inputs, trainingData.outputs);
            }

            result = nn.Predict(new double[] { 0, 1 });
            System.Console.WriteLine($"After: {result}");
        }
    }
}
