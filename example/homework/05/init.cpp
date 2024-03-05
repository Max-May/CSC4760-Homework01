#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <stdio.h>

// Create a program that compares a parallel for loop and a standard for loop for summing rows of a View with Kokkos Timer.

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  // Make View and create values
  long int n = 1000000000;
  int out = 0;
  double time;

  Kokkos::View<int*> A("A", n);
  Kokkos::parallel_for("init", A.extent(0), KOKKOS_LAMBDA(const int& i){
    A(i) = i+1;
  });

  // sum loops
  Kokkos::Timer timer;
  for (long int i=0;i<n;i++){
    out += A(i);
  }
  // Output times
  time = timer.seconds();
  printf("Normal loop: %f seconds\n", time);

  out =0;
  timer.reset();
  Kokkos::parallel_reduce("parallel", A.extent(0),
    KOKKOS_LAMBDA(const long int& i, int& lsum){
      lsum += A(i);
    },
  out);
  time = timer.seconds();
  printf("Parallel reduce loop: %f seconds\n", time);
  }
  Kokkos::finalize();
}
