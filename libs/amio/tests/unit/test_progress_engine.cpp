#include <atomic>
#include <cassert>
#include <chrono>
#include <stdexcept>

#include "execution/progress_engine.hpp"

using namespace amio::detail;

int main() {
    ProgressEngine caller(ProgressPolicy::Caller, 2, 0);
    std::atomic<int> calls{0};
    assert(caller.submit([&] {
        ++calls;
        return AMIO_OK;
    }) == AMIO_OK);
    assert(caller.submit([&] {
        ++calls;
        return AMIO_OK;
    }) == AMIO_OK);
    assert(caller.submit([&] {
        ++calls;
        return AMIO_OK;
    }) == AMIO_ERR_QUEUE_FULL);
    assert(caller.close(std::chrono::milliseconds(0)) == AMIO_ERR_OPERATION_PENDING);
    std::size_t completed = 0;
    assert(caller.progress(1, completed) == AMIO_OK && completed == 1 && calls == 1);
    assert(caller.progress(8, completed) == AMIO_OK && completed == 1 && calls == 2);
    assert(caller.close(std::chrono::milliseconds(10)) == AMIO_OK);
    assert(caller.submit([] { return AMIO_OK; }) == AMIO_ERR_INVALID_POLICY);

    ProgressEngine failing(ProgressPolicy::Caller, 2, 0);
    assert(failing.submit([] { return AMIO_ERR_BACKEND_FAILURE; }) == AMIO_OK);
    assert(failing.submit([] { return AMIO_OK; }) == AMIO_OK);
    assert(failing.progress(2, completed) == AMIO_ERR_BACKEND_FAILURE && completed == 2);
    assert(failing.close(std::chrono::milliseconds(10)) == AMIO_ERR_BACKEND_FAILURE);

    bool rejected = false;
    try {
        ProgressEngine invalid(ProgressPolicy::Background, 2, 2);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    ProgressEngine background(ProgressPolicy::Background, 2, 3);
    std::atomic<bool> ran{false};
    assert(background.submit([&] {
        ran = true;
        return AMIO_OK;
    }) == AMIO_OK);
    assert(background.close(std::chrono::seconds(1)) == AMIO_OK);
    assert(ran);
}
