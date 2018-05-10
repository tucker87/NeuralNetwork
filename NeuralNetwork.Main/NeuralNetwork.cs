using System;
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
        public double LearningRate { get; set; } = 0.2;

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

        public void Train(double[] inputsArray, double[] targetsArray, bool debugFlag = false)
        {
            Debug($"Training: {inputsArray[0]} {inputsArray[1]} | {targetsArray[0]}", debugFlag);
            var inputs = M.DenseOfColumnArrays(inputsArray);
            var hiddens = CalculateHiddenLayer(inputs);
            var outputs = CalculateOutputLayer(hiddens);
            Debug($"Output: {outputs[0, 0]:F6}", debugFlag);
            var targets = M.DenseOfColumnArrays(targetsArray);
            var outputErrors = targets - outputs;

            var outputGradients = outputs.Map(dSigma).PointwiseMultiply(outputErrors) * LearningRate;
            Debug($"Output Gradients: {outputGradients[0, 0]:F6}", debugFlag);

            var weightHODeltas = outputGradients * hiddens.Transpose();

            Debug($"HO Weights Before: {Weights_HO[0,0]}", debugFlag);
            Weights_HO = Weights_HO + weightHODeltas;
            Debug($"HO Weights After: {Weights_HO[0, 0]}", debugFlag);


            if (double.IsNaN(Weights_HO[0, 0]))
                throw new Exception("WTF!");

            Bias_O = Bias_O + outputGradients;

            var hiddenErrors = Weights_HO.Transpose() * outputErrors;

            var hiddenGradients = hiddens.Map(dSigma).PointwiseMultiply(hiddenErrors) * LearningRate;

            var wightIHDeltas = hiddenGradients * inputs.Transpose();

            Weights_IH = Weights_IH + wightIHDeltas;
            Bias_H = Bias_H + hiddenGradients;
        }

        private Matrix<double> CalculateHiddenLayer(Matrix<double> inputs)
        {
            return (Weights_IH * inputs + Bias_H).Map(MathNet.Numerics.SpecialFunctions.Logistic);
        }

        private Matrix<double> CalculateOutputLayer(Matrix<double> hiddens)
        {
            return (Weights_HO * hiddens + Bias_O).Map(MathNet.Numerics.SpecialFunctions.Logistic);
        }

        private double dSigma(double x)
        {
            return x * (1 - x);
        }

        private void Debug(string message, bool flag)
        {
            if(flag)
                Console.WriteLine(message);
        }
    }
}
