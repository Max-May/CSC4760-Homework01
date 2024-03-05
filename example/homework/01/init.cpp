#include <Kokkos_Core.hpp>
#include <cstdio>
#include <stdio.h>
#include <iostream>
// Problem: Link and run program with Kokkos where you initialize a View and print out its name with the $.label()$ method.

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  // Make View
  Kokkos::View<int*> A("Ex01", 1);

  // print name
  std::cout << "My label: " << A.label() << std::endl;
  }
  Kokkos::finalize();
}
