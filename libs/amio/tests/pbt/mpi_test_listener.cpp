// mpi_test_listener.cpp -- Catch2 listener that initializes MPI for the
// duration of a PBT executable.
//
// Several property-based tests open real backend datasets via
// amio_open_dataset.  The NetCDF-4 backend issues parallel HDF5 calls
// (nc_create_par / nc_open_par) that require MPI to be initialized by the
// host application BEFORE the driver is opened.  In the AMIO contract the
// host owns MPI initialization; for a test executable that "host" is the
// test binary itself.
//
// This listener initializes MPI (at MPI_THREAD_MULTIPLE, matching the
// background-MPI-IO requirement) once when the test run starts, and
// finalizes it when the run ends.  Linking this translation unit into a
// PBT target is sufficient to register the listener -- no test code needs
// to change.
//
// Guarded with MPI_Initialized / MPI_Finalized so it is safe even if some
// other component already initialized MPI.

#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

// Use only the MPI C API; suppress the deprecated C++ MPI bindings so we
// do not need to link libmpi_cxx.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

namespace {

class MpiLifecycleListener : public Catch::EventListenerBase {
   public:
    using Catch::EventListenerBase::EventListenerBase;

    void testRunStarting(const Catch::TestRunInfo &) override {
        int already = 0;
        MPI_Initialized(&already);
        if (!already) {
            int provided = 0;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
            owns_mpi_ = true;
        }
    }

    void testRunEnded(const Catch::TestRunStats &) override {
        if (!owns_mpi_) {
            return;
        }
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    }

   private:
    bool owns_mpi_ = false;
};

}  // namespace

CATCH_REGISTER_LISTENER(MpiLifecycleListener)
