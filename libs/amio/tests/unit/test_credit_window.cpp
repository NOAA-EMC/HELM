#include <cassert>

#include "execution/credit_window.hpp"

using namespace amio::detail;

int main() {
    CreditWindow window({2, 1024, 2, 2048, 4, 256, 2});
    auto send = window.acquire(0, CreditOwner::Sender, 1024, 2);
    assert(send.has_value());
    assert(!window.acquire(0, CreditOwner::Sender, 1, 1).has_value());
    auto receive = window.acquire(0, CreditOwner::Receiver, 1024, 2);
    assert(receive.has_value());                                          // sender and receiver copies coexist in total accounting
    assert(!window.acquire(1, CreditOwner::Receiver, 1, 1).has_value());  // global data budget exhausted
    auto control = window.acquire_control(128, 1);
    assert(control.has_value());  // control plane stays live under data-credit exhaustion
    auto altered = *control;
    ++altered.bytes;
    assert(!window.release(altered));  // a forged or corrupted ticket cannot release another reservation
    auto snapshot = window.snapshot();
    assert(snapshot.sender_bytes == 1024 && snapshot.receiver_bytes == 1024);
    assert(snapshot.control_bytes == 128);
    assert(window.release(*send));
    assert(!window.release(*send));
    assert(window.release(*receive));
    assert(window.release(*control));
    snapshot = window.snapshot();
    assert(snapshot.data_bytes == 0 && snapshot.data_descriptors == 0);
    assert(snapshot.control_bytes == 0 && snapshot.control_descriptors == 0);
    const auto peak = window.high_water();
    assert(peak.sender_bytes == 1024 && peak.receiver_bytes == 1024 && peak.data_bytes == 2048);
    assert(peak.sender_descriptors == 2 && peak.receiver_descriptors == 2 && peak.data_descriptors == 4);
    assert(peak.control_bytes == 128 && peak.control_descriptors == 1);
}
