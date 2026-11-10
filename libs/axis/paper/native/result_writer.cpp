// SPDX-License-Identifier: Apache-2.0
#include "result_writer.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
namespace paper {
std::string quote(const std::string &s) {
    std::ostringstream o;
    o << '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                o << "\\\"";
                break;
            case '\\':
                o << "\\\\";
                break;
            case '\n':
                o << "\\n";
                break;
            case '\r':
                o << "\\r";
                break;
            case '\t':
                o << "\\t";
                break;
            default:
                if (c < 32)
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
                      << std::dec;
                else
                    o << c;
        }
    }
    o << '"';
    return o.str();
}
std::string number(double x) {
    if (!std::isfinite(x))
        return "null";
    std::ostringstream o;
    o << std::setprecision(17) << x;
    return o.str();
}
void atomic_text(const std::filesystem::path &p, const std::string &s) {
    auto t = p.string() + ".tmp";
    {
        std::ofstream f(t);
        f << s << '\n';
        if (!f)
            throw Error(5, "Cannot write " + t);
    }
    std::filesystem::rename(t, p);
}
void observations(const std::filesystem::path &p, const std::vector<Observation> &v) {
    std::ostringstream o;
    o << "repetition,phase,applications,seconds,status\n";
    for (auto &r : v)
        o << r.repetition << ',' << r.phase << ',' << r.applications << ',' << std::setprecision(17)
          << r.seconds << ',' << r.status << '\n';
    atomic_text(p, o.str());
}
int guarded(const std::function<void()> &fn) {
    try {
        fn();
        return 0;
    } catch (const Error &e) {
        std::cerr << e.what() << '\n';
        return e.code;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 5;
    }
}
}  // namespace paper
