#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>

// Declare a 5 ∗ 7 ∗ 12 ∗ n View

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
  // Make View
  int n = 10;
  Kokkos::View<int*[5][7][12]> A("A", n);

  // print name
  std::cout << "Label view A: " << A.label() << std::endl;
  for (int i=0;i<4;i++){
    std::cout << "Dim[" << i << "] = "  << A.extent(i);
    std::cout << std::endl;
    }
  }
  Kokkos::finalize();
  return 0;
}
