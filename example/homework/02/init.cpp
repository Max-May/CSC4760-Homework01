#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>

// Problem: Make an n ∗ m View where each index equals 1000 ∗ i ∗ j

int main(int argc, char* argv[]) {
	Kokkos::initialize(argc, argv);
	{
  	// set n and m, you can change these values
  	int n,m;
	n = 16;
	m = 16;

  	// Make View
  	Kokkos::View<int**> A("A", n, m);

  	// set values to 1000 * i * j;
  	Kokkos::parallel_for("Loop1", A.extent(0), KOKKOS_LAMBDA (const int i) {
    		// accessor method uses parentheses
    		Kokkos::parallel_for("Loop2", A.extent(1), KOKKOS_LAMBDA (const int j) {
      			A(i, j) = 1000 * i * j;
    			});
  		});

	for (int i=0;i<n;i++){
		for (int j=0;j<m;j++){
  			std::cout << A(i,j) << " ";
			}
		std::cout << std::endl;
		}
  	}
  	Kokkos::finalize();
  	return 0;
}

