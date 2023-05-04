//
//  main.cpp
//  NeuralNetwork
//
//  Created by Santiago Becerra on 9/15/19.
//  Copyright © 2019 Santiago Becerra. All rights reserved.
//
//

#include <iostream>
#include <list>
#include <cstdlib>
#include <math.h>
#include <random>


// Simple network that can learn XOR
// Feartures : sigmoid activation function, stochastic gradient descent, and mean square error fuction

// Potential improvements :
// Different activation functions
// Batch training
// Different error funnctions
// Arbitrary number of hidden layers
// Read training end test data from a file
// Add visualization of training
// Add recurrence? (maybe that should be a separate project)

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }
double init_weight() { return ((double)rand())/((double)RAND_MAX); }
void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int main(int argc, const char * argv[]) {

/*
    static const int numInputs = 2;
    static const int numHiddenNodes = 2;
    static const int numOutputs = 1;
*/




	// Getting these vars from commandline
    int numInputs = atoi(argv[1]);
    int numHiddenNodes = atoi(argv[2]);
    int numOutputs = atoi(argv[3]);
    int numTrainingSets = atoi(argv[4]);
    
    const double lr = 0.1f;
    
	Format  rm({Dense,Dense});
	Format  dv({Dense});
    /*double X[numInputs];
    double Y[numHiddenNodes];
    */
    double X[numInputs];
    Tensor<double> taco_X({numInputs},dv);
    double Y[numHiddenNodes];
    //double Y[numHiddenNodes];
    Tensor<double> taco_Y({numHiddenNodes},dv);

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];
    
    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    Tensor<double> taco_hiddenWeights({numInputs,numHiddenNodes},rm);
    double outputWeights[numHiddenNodes][numOutputs];
    
    // use random and cast to double

   // Generating random values form the uniform distribution
  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  
    double training_inputs[numTrainingSets][numInputs];
  for (int i = 0; i < numTrainingSets; ++i) {
    for (int j = 0; j < numInputs; ++j) {
      training_inputs[i][j] = unif(gen);
    }
  }

    double training_outputs[numTrainingSets][numOutputs];
  for (int i = 0; i < numTrainingSets; ++i) {
    for (int j = 0; j < numOutputs; ++j) {
      training_inputs[i][j] = unif(gen);
    }
  }
  // End of random val generation 

    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weight();
        }
    }
    for (int i=0; i<numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j=0; j<numOutputs; j++) {
            outputWeights[i][j] = init_weight();
        }
    }
    for (int i=0; i<numOutputs; i++) {
        outputLayerBias[i] = init_weight();
    }
    
    int trainingSetOrder[] = {0,1,2,3};
    
    for (int n=0; n < 10000; n++) {
       // shuffle(trainingSetOrder,numTrainingSets);
        for (int x=0; x<numTrainingSets; x++) {
            
           // int i = trainingSetOrder[x];
            int i = x;
            
            // Forward pass
            
           /* for (int j=0; j<numHiddenNodes; j++) {
                double activation=hiddenLayerBias[j];
                 for (int k=0; k<numInputs; k++) {
                    activation+=training_inputs[i][k]*hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            */

	    for (int k = 0; k < numInputs; k++){
		    X[k] = training_inputs[i][k];
	    }
	    
	    /*for (int j =0; j < numHiddenNodes; j++){
		    Y[j] = 0.0f;
	    }*/
	    for (int j =0; j < numHiddenNodes; j++){
		    taco_Y.insert({j},0.0f);
	    }
	    
	    for(int j = 0; j < numHiddenNodes; j++){
                    for(int k = 0; k < numInputs; k++){
                            taco_hiddenWeights.insert({k,j},hiddenWeights[k][j]);
                    }
            }
	    // MAT_VEC

	    for(int j = 0; j < numHiddenNodes; j++){
		    for(int k = 0; k < numInputs; k++){
			    taco_Y(j) = (hiddenWeights(k,j) * X(k));
		    }
	    }
	    
	    //Copying the results to Y
	    for(int j = 0; j < numHiddenNodes; j++){
                            Y[j] = taco_Y(j); 
            }


	    for(int j = 0; j < numHiddenNodes; j++){
		    hiddenLayer[j] = sigmoid(Y[i] + hiddenLayerBias[j]);
	    }
	    // End here 
            for (int j=0; j<numOutputs; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<numHiddenNodes; k++) {
                    activation+=hiddenLayer[k]*outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }
            
            std::cout << "Input:" << training_inputs[i][0] << " " << training_inputs[i][1] << "    Output:" << outputLayer[0] << "    Expected Output: " << training_outputs[i][0] << "\n";
            
           // Backprop
            
            double deltaOutput[numOutputs];
            for (int j=0; j<numOutputs; j++) {
                double errorOutput = (training_outputs[i][j]-outputLayer[j]);
                deltaOutput[j] = errorOutput*dSigmoid(outputLayer[j]);
            }
            
            double deltaHidden[numHiddenNodes];
            for (int j=0; j<numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<numOutputs; k++) {
                    errorHidden+=deltaOutput[k]*outputWeights[j][k];
                }
                deltaHidden[j] = errorHidden*dSigmoid(hiddenLayer[j]);
            }
            
            for (int j=0; j<numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j]*lr;
                for (int k=0; k<numHiddenNodes; k++) {
                    outputWeights[k][j]+=hiddenLayer[k]*deltaOutput[j]*lr;
                }
            }
            
            for (int j=0; j<numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j]*lr;
                for(int k=0; k<numInputs; k++) {
                    hiddenWeights[k][j]+=training_inputs[i][k]*deltaHidden[j]*lr;
                }
            }
        }
    }
    
    // Print weights
    std::cout << "Final Hidden Weights\n[ ";
    for (int j=0; j<numHiddenNodes; j++) {
        std::cout << "[ ";
        for(int k=0; k<numInputs; k++) {
            std::cout << hiddenWeights[k][j] << " ";
        }
        std::cout << "] ";
    }
    std::cout << "]\n";
    
    std::cout << "Final Hidden Biases\n[ ";
    for (int j=0; j<numHiddenNodes; j++) {
        std::cout << hiddenLayerBias[j] << " ";

    }
    std::cout << "]\n";
    std::cout << "Final Output Weights";
    for (int j=0; j<numOutputs; j++) {
        std::cout << "[ ";
        for (int k=0; k<numHiddenNodes; k++) {
            std::cout << outputWeights[k][j] << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "Final Output Biases\n[ ";
    for (int j=0; j<numOutputs; j++) {
        std::cout << outputLayerBias[j] << " ";
        
    }
    std::cout << "]\n";

    return 0;
}

