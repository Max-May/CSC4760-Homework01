/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <cstdio>

#include <Kokkos_Core.hpp>

namespace Test {

namespace {

template <class ExecSpace, class ScheduleType>
struct TestRange {
  using value_type = int;  ///< alias required for the parallel_reduce

  using view_type = Kokkos::View<value_type *, ExecSpace>;

  view_type m_flags;
  view_type result_view;

  struct VerifyInitTag {};
  struct ResetTag {};
  struct VerifyResetTag {};
  struct OffsetTag {};
  struct VerifyOffsetTag {};

  int N;
#ifndef KOKKOS_WORKAROUND_OPENMPTARGET_GCC
  static const int offset = 13;
#else
  int offset;
#endif
  TestRange(const size_t N_)
      : m_flags(Kokkos::view_alloc(Kokkos::WithoutInitializing, "flags"), N_),
        result_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, "results"),
                    N_),
        N(N_) {
#ifdef KOKKOS_WORKAROUND_OPENMPTARGET_GCC
    offset = 13;
#endif
  }

  void test_for() {
    typename view_type::HostMirror host_flags =
        Kokkos::create_mirror_view(m_flags);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, ScheduleType>(0, N),
                         *this);

    {
      using ThisType = TestRange<ExecSpace, ScheduleType>;
      std::string label("parallel_for");
      Kokkos::Impl::ParallelConstructName<ThisType, void> pcn(label);
      ASSERT_EQ(pcn.get(), label);
      std::string empty_label("");
      Kokkos::Impl::ParallelConstructName<ThisType, void> empty_pcn(
          empty_label);
      ASSERT_EQ(empty_pcn.get(), typeid(ThisType).name());
    }

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace, ScheduleType, VerifyInitTag>(0, N),
        *this);

    {
      using ThisType = TestRange<ExecSpace, ScheduleType>;
      std::string label("parallel_for");
      Kokkos::Impl::ParallelConstructName<ThisType, VerifyInitTag> pcn(label);
      ASSERT_EQ(pcn.get(), label);
      std::string empty_label("");
      Kokkos::Impl::ParallelConstructName<ThisType, VerifyInitTag> empty_pcn(
          empty_label);
      ASSERT_EQ(empty_pcn.get(), std::string(typeid(ThisType).name()) + "/" +
                                     typeid(VerifyInitTag).name());
    }

    Kokkos::deep_copy(host_flags, m_flags);

    int error_count = 0;
    for (int i = 0; i < N; ++i) {
      if (int(i) != host_flags(i)) ++error_count;
    }
    ASSERT_EQ(error_count, int(0));

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace, ScheduleType, ResetTag>(0, N), *this);
    Kokkos::parallel_for(
        std::string("TestKernelFor"),
        Kokkos::RangePolicy<ExecSpace, ScheduleType, VerifyResetTag>(0, N),
        *this);

    Kokkos::deep_copy(host_flags, m_flags);

    error_count = 0;
    for (int i = 0; i < N; ++i) {
      if (int(2 * i) != host_flags(i)) ++error_count;
    }
    ASSERT_EQ(error_count, int(0));

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace, ScheduleType, OffsetTag>(offset,
                                                                N + offset),
        *this);
    Kokkos::parallel_for(
        std::string("TestKernelFor"),
        Kokkos::RangePolicy<ExecSpace, ScheduleType, VerifyOffsetTag>(0, N),
        *this);

    Kokkos::deep_copy(host_flags, m_flags);

    error_count = 0;
    for (int i = 0; i < N; ++i) {
      if (i + offset != host_flags(i)) ++error_count;
    }
    ASSERT_EQ(error_count, int(0));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const { m_flags(i) = i; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const VerifyInitTag &, const int i) const {
    if (i != m_flags(i)) {
      // FIXME_SYCL printf needs a workaround
#ifndef __SYCL_DEVICE_ONLY__
      printf("TestRange::test_for_error at %d != %d\n", i, m_flags(i));
#endif
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ResetTag &, const int i) const {
    m_flags(i) = 2 * m_flags(i);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const VerifyResetTag &, const int i) const {
    if (2 * i != m_flags(i)) {
      // FIXME_SYCL printf needs a workaround
#ifndef __SYCL_DEVICE_ONLY__
      printf("TestRange::test_for_error at %d != %d\n", i, m_flags(i));
#endif
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const OffsetTag &, const int i) const {
    m_flags(i - offset) = i;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const VerifyOffsetTag &, const int i) const {
    if (i + offset != m_flags(i)) {
      // FIXME_SYCL printf needs a workaround
#ifndef __SYCL_DEVICE_ONLY__
      printf("TestRange::test_for_error at %d != %d\n", i + offset, m_flags(i));
#endif
    }
  }

  //----------------------------------------

  void test_reduce() {
    value_type total = 0;

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, ScheduleType>(0, N),
                         *this);

    Kokkos::parallel_reduce("TestKernelReduce",
                            Kokkos::RangePolicy<ExecSpace, ScheduleType>(0, N),
                            *this, total);
    // sum( 0 .. N-1 )
    ASSERT_EQ(size_t((N - 1) * (N) / 2), size_t(total));

    Kokkos::parallel_reduce(
        "TestKernelReduce_long",
        Kokkos::RangePolicy<ExecSpace, ScheduleType, long>(0, N), *this, total);
    // sum( 0 .. N-1 )
    ASSERT_EQ(size_t((N - 1) * (N) / 2), size_t(total));

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecSpace, ScheduleType, OffsetTag>(offset,
                                                                N + offset),
        *this, total);
    // sum( 1 .. N )
    ASSERT_EQ(size_t((N) * (N + 1) / 2), size_t(total));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &update) const {
    update += m_flags(i);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const OffsetTag &, const int i, value_type &update) const {
    update += 1 + m_flags(i - offset);
  }

  //----------------------------------------

  void test_scan() {
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, ScheduleType>(0, N),
                         *this);

    auto check_scan_results = [&]() {
      auto const host_mirror =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result_view);
      for (int i = 0; i < N; ++i) {
        if (((i + 1) * i) / 2 != host_mirror(i)) {
          std::cout << "Error at " << i << std::endl;
          EXPECT_EQ(size_t(((i + 1) * i) / 2), size_t(host_mirror(i)));
        }
      }
    };

    Kokkos::parallel_scan(
        "TestKernelScan",
        Kokkos::RangePolicy<ExecSpace, ScheduleType, OffsetTag>(0, N), *this);

    check_scan_results();

    value_type total = 0;
    Kokkos::parallel_scan(
        "TestKernelScanWithTotal",
        Kokkos::RangePolicy<ExecSpace, ScheduleType, OffsetTag>(0, N), *this,
        total);

    check_scan_results();

    ASSERT_EQ(size_t((N - 1) * (N) / 2), size_t(total));  // sum( 0 .. N-1 )
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const OffsetTag &, const int i, value_type &update,
                  bool final) const {
    update += m_flags(i);

    if (final) {
      if (update != (i * (i + 1)) / 2) {
#ifndef __SYCL_DEVICE_ONLY__
        printf("TestRange::test_scan error (%d,%d) : %d != %d\n", i, m_flags(i),
               (i * (i + 1)) / 2, update);
#endif
      }
      result_view(i) = update;
    }
  }

  void test_dynamic_policy() {
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    auto const N_no_implicit_capture = N;
    using policy_t =
        Kokkos::RangePolicy<ExecSpace, Kokkos::Schedule<Kokkos::Dynamic> >;

    {
      Kokkos::View<size_t *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic> >
          count("Count", ExecSpace::concurrency());
      Kokkos::View<int *, ExecSpace> a("A", N);

      Kokkos::parallel_for(
          policy_t(0, N), KOKKOS_LAMBDA(const int &i) {
            for (int k = 0; k < (i < N_no_implicit_capture / 2 ? 1 : 10000);
                 k++) {
              a(i)++;
            }
            count(ExecSpace::impl_hardware_thread_id())++;
          });

      int error = 0;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<ExecSpace>(0, N),
          KOKKOS_LAMBDA(const int &i, value_type &lsum) {
            lsum += (a(i) != (i < N_no_implicit_capture / 2 ? 1 : 10000));
          },
          error);
      ASSERT_EQ(error, 0);

      if ((ExecSpace::concurrency() > (int)1) &&
          (N > static_cast<int>(4 * ExecSpace::concurrency()))) {
        size_t min = N;
        size_t max = 0;
        for (int t = 0; t < ExecSpace::concurrency(); t++) {
          if (count(t) < min) min = count(t);
          if (count(t) > max) max = count(t);
        }
        ASSERT_TRUE(min < max);

        // if ( ExecSpace::concurrency() > 2 ) {
        //  ASSERT_TRUE( 2 * min < max );
        //}
      }
    }

    {
      Kokkos::View<size_t *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic> >
          count("Count", ExecSpace::concurrency());
      Kokkos::View<int *, ExecSpace> a("A", N);

      value_type sum = 0;
      Kokkos::parallel_reduce(
          policy_t(0, N),
          KOKKOS_LAMBDA(const int &i, value_type &lsum) {
            for (int k = 0; k < (i < N_no_implicit_capture / 2 ? 1 : 10000);
                 k++) {
              a(i)++;
            }
            count(ExecSpace::impl_hardware_thread_id())++;
            lsum++;
          },
          sum);
      ASSERT_EQ(sum, N);

      int error = 0;
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<ExecSpace>(0, N),
          KOKKOS_LAMBDA(const int &i, value_type &lsum) {
            lsum += (a(i) != (i < N_no_implicit_capture / 2 ? 1 : 10000));
          },
          error);
      ASSERT_EQ(error, 0);

      if ((ExecSpace::concurrency() > (int)1) &&
          (N > static_cast<int>(4 * ExecSpace::concurrency()))) {
        size_t min = N;
        size_t max = 0;
        for (int t = 0; t < ExecSpace::concurrency(); t++) {
          if (count(t) < min) min = count(t);
          if (count(t) > max) max = count(t);
        }
        ASSERT_TRUE(min < max);

        // if ( ExecSpace::concurrency() > 2 ) {
        //  ASSERT_TRUE( 2 * min < max );
        //}
      }
    }
#endif
  }
};

}  // namespace

TEST(TEST_CATEGORY, range_for) {
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> > f(0);
    f.test_for();
  }
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(0);
    f.test_for();
  }

  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> > f(2);
    f.test_for();
  }
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(3);
    f.test_for();
  }

  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> > f(1000);
    f.test_for();
  }
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(1001);
    f.test_for();
  }
}

TEST(TEST_CATEGORY, range_reduce) {
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> > f(0);
    f.test_reduce();
  }
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(0);
    f.test_reduce();
  }

  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> > f(2);
    f.test_reduce();
  }
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(3);
    f.test_reduce();
  }

  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> > f(1000);
    f.test_reduce();
  }
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(1001);
    f.test_reduce();
  }
}

// FIXME_SYCL needs parallel_scan
#ifndef KOKKOS_ENABLE_SYCL
#ifndef KOKKOS_ENABLE_OPENMPTARGET
TEST(TEST_CATEGORY, range_scan) {
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> > f(0);
    f.test_scan();
  }
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(0);
    f.test_scan();
  }
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && \
    !defined(KOKKOS_ENABLE_SYCL)
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(0);
    f.test_dynamic_policy();
  }
#endif

  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> > f(2);
    f.test_scan();
  }
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(3);
    f.test_scan();
  }
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && \
    !defined(KOKKOS_ENABLE_SYCL)
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(3);
    f.test_dynamic_policy();
  }
#endif

  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> > f(1000);
    f.test_scan();
  }
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(1001);
    f.test_scan();
  }
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && \
    !defined(KOKKOS_ENABLE_SYCL)
  {
    TestRange<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> > f(1001);
    f.test_dynamic_policy();
  }
#endif
}
#endif
#endif
}  // namespace Test
