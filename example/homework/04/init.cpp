#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>

// Do simple parallel reduce to output the maximum element in a View

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  // Make View and create values
  int n=20;
  int output=0;
  Kokkos::View<int*> A("A", n);

  for (int i=0;i<n;i++){
    A(i) = i*2;
    }

  // Do a parallel reduction
  Kokkos::parallel_reduce("Maximum reduce",
    A.extent(0),
    KOKKOS_LAMBDA(const int& i, int& max_val){
      if (max_val < A(i)) max_val = A(i);
    }, Kokkos::Max<int>(output));

  for (int i=0;i<n;i++){
    std::cout << A(i) << " ";
    }
  std::cout << std::endl;
  std::cout << "Maximum result: " << output << std::endl;
  }
  Kokkos::finalize();
}
