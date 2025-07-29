#include <iostream>
using namespace std;

int main() {
  const int N = 5;
  float * array = new float[N * N];

  for(int i = 0; i < N * N; i++) {
    array[i] = i;
  }

  //float * mat = array;
  //
  //for(int i = 0; i < N; i++) {
  //  float * row = &mat[N * i];
  //    
  //  for(int j = 0; j < N; j++) {
  //    cout << "i=" << i << "\t" << "j=" << j << "\t" << row[j] << endl; 
  //  }
  //}
 
  for(int i = 0; i < N; i++) {
    float * row = array + N * i;
      
    for(int j = 0; j < N; j++) {
      cout << "i=" << i << "\t" << "j=" << j << "\t" << row[j] << endl; 
    }
  }
}
