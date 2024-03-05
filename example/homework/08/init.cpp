#include <Kokkos_Core.hpp>
#include <cstdio>
#include <stdio.h>

// Create a program that does matrix multiply between a 2D View and a 1D View with at least one loop of parallelism.
// For a test case:
// a = [130, 137, 115]   b = [221]
//     [224, 158, 187]       [12]
//     [ 54, 211, 120]       [157]
// Extra credit: make a function and check for correct shape/dimensions
struct MatrixMult {
  MatrixMult (const Kokkos::View<int**>& x, const Kokkos::View<int**>& y, const Kokkos::View<int**>& z):
    x_ (x), y_ (y), z_ (z) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const{
    Kokkos::parallel_for("Loop2", y_.extent(1), KOKKOS_LAMBDA(const int& j){
      Kokkos::parallel_for("Loop3", x_.extent(1), KOKKOS_LAMBDA(const int& k){
        z_(i,j) += x_(i,k) * y_(k,j);
      });
      printf("Z[%d,%d]: %d\n", i, j, z_(i,j));
    });
  }

  private:
    Kokkos::View<int**> x_;
    Kokkos::View<int**> y_;
    Kokkos::View<int**> z_;
};

bool checkDim(Kokkos::View<int**>& a, Kokkos::View<int**>& b){
  bool dimCheck;

  dimCheck = a.extent(1) == b.extent(0);
  return dimCheck;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  int n,m,k;

  if (argc > 3){
  n = atoi(argv[1]);
  m = atoi(argv[2]);
  k = atoi(argv[3]);
  } else{
  n = 3;
  m = 3;
  k = 1;
  }


  // Make View and add values
  Kokkos::View<int**> A("A", n, m);
  Kokkos::View<int**> B("B", m, k);
  Kokkos::View<int**> C("C", n, k);

  bool cont = checkDim(A, B);
  if (!cont) {
    printf("Dimensions of A:[%d,%d] and B:[%d,%d] don't line up!", A.extent(0), A.extent(1), B.extent(0), B.extent(1));
    exit(-1);
  }

  for (int i=0;i<n;i++){
    for (int j=0;j<m;j++){
      A(i,j) = i + j;
      }
    }

  for (int i=0;i<m;i++){
    for (int j=0;j<k;j++){
      B(i,j) = i + j;
    }
  }

  for (int i=0;i<n;i++){
    for (int j=0;j<k;j++){
      C(i,j) = 0;
    }
  }

  // Do a matrix multiply
  //MatrixMult mult(A, B, C);
  //Kokkos::parallel_for("Loop1", A.extent(0), mult);
  Kokkos::parallel_for("Loop1", A.extent(0), KOKKOS_LAMBDA(const int& i){
    Kokkos::parallel_for("Loop2", B.extent(1), KOKKOS_LAMBDA(const int& j){
      Kokkos::parallel_for("Loop3", A.extent(1), KOKKOS_LAMBDA(const int& k){
        C(i,j) += A(i,k) * B(k,j);
        });
      });
    });
  // Output addition
  printf("A:\n");
  for (int i=0;i<n;i++){
    for (int j=0;j<m;j++){
      printf("%d ", A(i,j));
    }
    printf("\n");
  }
  printf("\nB:\n");
  for (int i=0;i<m;i++){
    for (int j=0;j<k;j++){
      printf("%d ", B(i,j));
    }
    printf("\n");
  }
  printf("\nC:\n");
  for (int i=0;i<n;i++){
    for (int j=0;j<k;j++){
      printf("%d ", C(i,j));
    }
    printf("\n");
  }
  }
  Kokkos::finalize();
}
