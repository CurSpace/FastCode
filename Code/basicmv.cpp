#include <chrono>
#include <math.h>
#include <iostream>
#include <random>
#define r 1024 
#define c 1024 
int main(){
	//std::default_random_engine gen(0); 
 	//std::uniform_real_distribution<double> unif(0.0, 1.0);
	double A[r][c],x[c],C[r];

	// Generating a random matrix 
	
	for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
      //A[i][j] = unif(gen);
      A[i][j] = (double)(i + j);
    }
  }
	// Generatin a random vector
	for (int i = 0; i < r; ++i) {
    //x[i] = unif(gen);
    x[i] = (double)(i);
  }

	for(int i = 0; i < r; ++i){

		C[i] = 0.0;
	}
 // A[i][j] * x[i]
for(int i = 0;i < r; ++i){
       for(int j = 0;j < c; ++j){

	   std::cout<<i<<"\t"<<j<<std::endl;
           C[i] += (A[i][j] * x[j]);
       }
   }

auto end = std::chrono::steady_clock::now();

double sum = 0.0;
for(int i = 0; i < r; ++i){
	sum += C[i];
	
}
std::cout << sum;
std::cout << "Elapsed time in nanoseconds: "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
        << " ns" << std::endl;
}
