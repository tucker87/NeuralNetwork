using MathNet.Numerics.LinearAlgebra;

// ReSharper disable InconsistentNaming

namespace NeuralNetworkNS.Main
{
    public class NeuralNetwork
    {
        private readonly MatrixBuilder<double> M = Matrix<double>.Build;
        public Matrix<double> Weights_IH { get; set; }
        public Matrix<double> Weights_HO { get; set; }
        public Matrix<double> Bias_H { get; set; }
        public Matrix<double> Bias_O { get; set; }
        public double LearningRate { get; set; } = 0.1;

        public NeuralNetwork(int inputNodeCount, int hiddenNodeCount, int outputNodeCount)
        {
            Weights_IH = M.Random(hiddenNodeCount, inputNodeCount);
            Weights_HO = M.Random(outputNodeCount, hiddenNodeCount);
            Bias_H = M.Random(hiddenNodeCount, 1);
            Bias_O = M.Random(outputNodeCount, 1);
        }

        public double[] Predict(double[] inputsArray)
        {
            var inputs = M.DenseOfColumnArrays(inputsArray);
            var hiddens = CalculateHiddenLayer(inputs);
            var output = CalculateOutputLayer(hiddens);

            return output.ToRowMajorArray();
        }

        public void Train(double[] inputsArray, double[] targetsArray)
        {
            var inputs = M.DenseOfColumnArrays(inputsArray);
            var hiddens = CalculateHiddenLayer(inputs);
            var outputs = CalculateOutputLayer(hiddens);

            var targets = M.DenseOfColumnArrays(targetsArray);
            var outputErrors = targets.Subtract(outputs);

            var outputGradients = outputs.Map(MathNet.Numerics.SpecialFunctions.Logit) * outputErrors * LearningRate;

            var weightHODeltas = outputGradients * hiddens.Transpose();

            Weights_HO = Weights_HO + weightHODeltas;
            Bias_O = Bias_O + outputGradients;

            var hiddenErrors = Weights_HO.Transpose() * outputErrors;

            var hiddenGradients = hiddens.Map(MathNet.Numerics.SpecialFunctions.Logit);
            hiddenGradients = hiddenGradients * hiddenErrors;
            hiddenGradients = hiddenGradients * LearningRate;

            var wightIHDeltas = hiddenGradients * inputs.Transpose();

            Weights_IH = Weights_IH + wightIHDeltas;
            Bias_H = Bias_H + hiddenGradients;
        }

        private Matrix<double> CalculateHiddenLayer(Matrix<double> inputs)
        {
            var hiddens = Weights_IH * inputs + Bias_H;
            hiddens = hiddens.Map(MathNet.Numerics.SpecialFunctions.Logistic);
            return hiddens;
        }

        private Matrix<double> CalculateOutputLayer(Matrix<double> hiddens)
        {
            var output = Weights_HO * hiddens + Bias_O;
            output = output.Map(MathNet.Numerics.SpecialFunctions.Logistic);
            return output;
        }
    }
}
