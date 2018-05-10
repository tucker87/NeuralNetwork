using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworkNS.Main;

namespace NeuralNetworkNS.Console
{
    class Program
    {
        public static int Resolution { get; set; } = 10;


        static void Main(string[] args)
        {
            var random = new Random();

            var nn = new NeuralNetwork(2, 2, 1);
            
            var data = new List<(double[] inputs, double[] outputs)>
            {
                (new double[] { 0, 0 }, new double[]{0}),
                (new double[] { 1, 1 }, new double[]{0}),
                (new double[] { 0, 1 }, new double[]{1}),
                (new double[] { 1, 0 }, new double[]{1})
            };

            foreach (var i in Enumerable.Range(0, 99999999))
            {
                var index = random.Next(0, 4);
                var (inputs, targets) = data[index];
                nn.Train(inputs, targets);
                var result = nn.Predict(new double[] { 0, 1 })[0];
                var result2 = nn.Predict(new double[] { 1, 1 })[0];
                Draw(nn);
            }
        }

        private static void Draw(NeuralNetwork nn)
        {
            for (var i = 0; i < Resolution; i++)
            {
                for (var j = 0; j < Resolution; j++)
                {
                    var x1 = (double) i / Resolution;
                    var x2 = (double) j / Resolution;
                    var inputs = new[] {x1, x2};
                    var y = nn.Predict(inputs)[0];
                    System.Console.ForegroundColor =
                        NearestConsoleColor.ClosestConsoleColor((byte) (y * 255), (byte) (y * 255), (byte) (y * 255));
                    System.Console.Write("i");
                }
                System.Console.Write("\r\n");
            }

            System.Console.SetCursorPosition(0, 0);
        }
    }
}
