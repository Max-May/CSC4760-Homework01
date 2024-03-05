#include <Kokkos_Core.hpp>
#include <cstdio>
#include <stdio.h>

struct PrefixSumFunctor {

  PrefixSumFunctor (const Kokkos::View<int*>& x) : x_ (x) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int&i, int& partial, bool is_final) const{
    Kokkos::Timer t_;
    partial += i;

    if (is_final){
      x_(i) = partial;
      double time = t_.seconds();
      printf("A[%d]: %d in: %.12lf seconds\n", i, x_(i), time);
      }
    }

  private:
    Kokkos::View<int*> x_;
};


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  int n = argc>1?atoi(argv[1]):10;
  int result;
  double time;

  Kokkos::Timer timer;
  Kokkos::View<int*> A("A", n);

  PrefixSumFunctor pfs(A);

  timer.reset();
  Kokkos::parallel_scan("Prefix sum", A.extent(0), pfs);

  time = timer.seconds();
  printf("\n\nTotal time: %.12lf seconds\n", time);
  }
  Kokkos::finalize();
}
