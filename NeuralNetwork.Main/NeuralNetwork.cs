using System;
using MathNet.Numerics.LinearAlgebra;

// ReSharper disable InconsistentNaming

namespace NeuralNetworkNS.Main
{
    public class NeuralNetwork
    {
        private readonly MatrixBuilder<double> M = Matrix<double>.Build;
        private Matrix<double> Weights_IH;
        private Matrix<double> Weights_HO;
        private Matrix<double> Bias_H;
        private Matrix<double> Bias_O;
        public double LearningRate { get; set; } = 0.3;

        public NeuralNetwork(int inputNodeCount, int hiddenNodeCount, int outputNodeCount)
        {
            Weights_IH = M.Random(hiddenNodeCount, inputNodeCount);
            Weights_HO = M.Random(outputNodeCount, hiddenNodeCount);
            Bias_H = M.Random(hiddenNodeCount, 1);
            Bias_O = M.Random(outputNodeCount, 1);
        }

        private (Matrix<double> inputs, Matrix<double> hiddens, Matrix<double> output) FeedForward(double[] inputsArray)
        {
            var inputs = M.DenseOfColumnArrays(inputsArray);
            var hiddens = CalculateHiddenLayer(inputs);
            var outputs = CalculateOutputLayer(hiddens);

            return (inputs, hiddens, outputs);
        }

        public double[] Predict(double[] inputsArray)
        {
            var (_,_,outputs) = FeedForward(inputsArray);

            return outputs.ToRowMajorArray();
        }

        public void Train(double[] inputsArray, double[] targetsArray, bool debugFlag = false)
        {
            var (inputs, hiddens, outputs) = FeedForward(inputsArray);
            var outputErrors = M.DenseOfColumnArrays(targetsArray) - outputs;
            AdjustLayer(outputErrors, outputs, hiddens, ref Weights_HO, ref Bias_O);
            var hiddenErrors = Weights_HO.Transpose() * outputErrors;
            AdjustLayer(hiddenErrors, hiddens, inputs, ref Weights_IH, ref Bias_H);
        }

        private void AdjustLayer(Matrix<double> outputErrors, Matrix<double> outputs, Matrix<double> hiddens, ref Matrix<double> weights, ref Matrix<double> bias)
        {
            var gradients = CalculateGradients(outputs, outputErrors);
            var deltas = CalculateDeltas(gradients, hiddens);
            weights = weights + deltas;
            bias = bias + gradients;
        }

        private Matrix<double> CalculateHiddenLayer(Matrix<double> inputs)
        {
            return (Weights_IH * inputs + Bias_H).Map(MathNet.Numerics.SpecialFunctions.Logistic);
        }

        private Matrix<double> CalculateOutputLayer(Matrix<double> hiddens)
        {
            return (Weights_HO * hiddens + Bias_O).Map(MathNet.Numerics.SpecialFunctions.Logistic);
        }

        private Matrix<double> CalculateGradients(Matrix<double> layer, Matrix<double> errors)
        {
            return layer.Map(dSigma).PointwiseMultiply(errors) * LearningRate;
        }

        private static Matrix<double> CalculateDeltas(Matrix<double> gradients, Matrix<double> feed)
        {
            return gradients * feed.Transpose();
        }

        private double dSigma(double x)
        {
            return x * (1 - x);
        }
    }
}
