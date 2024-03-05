#include <Kokkos_Core.hpp>
#include <cstdio>
#include <stdio.h>
#include <assert.h>

// Create a program that does matrix addition between a 2D View and a 1D View with at least one loop of parallelism.
// For a test case:
// a = [130, 137, 115]   b = [221]
//     [224, 158, 187]       [12]
//     [ 54, 211, 120]       [157]
// Extra credit: make a function and check for correct shape/dimensions

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  // Make View and add values
  Kokkos::View<int[3][3]> A("A");
  Kokkos::View<int[3]> B("B");
  Kokkos::View<int[3][3]> C("C");

  // Dimension A[0] and B[1] must be the same
  assert(A.extent(1) == B.extent(0));

  int temp = 0;
  for(int i=0;i<A.extent(0);i++){
    for(int j=0;j<A.extent(1);j++){
      A(i,j) = temp;
      temp++;
      }
    }
  for(int i=0;i<B.extent(0);i++){
    B(i) = (i+1)*3;
    }
  for(int i=0;i<C.extent(0);i++){
    for(int j=0;j<C.extent(1);j++){
      C(i,j) = 0;
      }
    }

  // Do a matrix add
  Kokkos::parallel_for("add", A.extent(0),
    KOKKOS_LAMBDA(const int& i){
    for(int j=0;j<A.extent(1);j++){
      C(i,j) = A(i,j) + B(j);
      }
    });

  // Output addition
  printf("A:\n");
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      printf("%d ", A(i,j));
      }
    printf("\n");
    }

  printf("\nB:\n");
  for(int i=0;i<3;i++){
    printf("%d\n", B(i));
    }

  printf("\nC:\n");
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      printf("%d ", C(i,j));
      }
    printf("\n");
    }
  }
  Kokkos::finalize();
  return 0;
}
