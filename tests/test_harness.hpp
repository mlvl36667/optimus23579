// Minimal assertion-based test harness.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace testing {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const std::string& name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

struct AssertionError {
    std::string msg;
};

inline void do_assert(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        throw AssertionError{std::string(file) + ":" + std::to_string(line) +
                             " assertion failed: " + expr};
    }
}

inline void do_assert_close(double a, double b, double tol, const char* file, int line) {
    if (std::fabs(a - b) > tol) {
        throw AssertionError{std::string(file) + ":" + std::to_string(line) +
                             " expected " + std::to_string(a) + " ~= " +
                             std::to_string(b)};
    }
}

inline int run_all() {
    int passed = 0, failed = 0;
    for (const auto& tc : registry()) {
        try {
            tc.fn();
            passed++;
        } catch (const AssertionError& e) {
            failed++;
            std::printf("[FAIL] %s\n       %s\n", tc.name.c_str(), e.msg.c_str());
        } catch (const std::exception& e) {
            failed++;
            std::printf("[FAIL] %s\n       exception: %s\n", tc.name.c_str(), e.what());
        }
    }
    std::printf("\n%d passed, %d failed, %d total\n", passed, failed,
                passed + failed);
    return failed == 0 ? 0 : 1;
}

}  // namespace testing

#define TEST(name) \
    static void name(); \
    static ::testing::Registrar _reg_##name(#name, name); \
    static void name()

#define CHECK(cond) ::testing::do_assert((cond), #cond, __FILE__, __LINE__)
#define CHECK_CLOSE(a, b, tol) ::testing::do_assert_close((a), (b), (tol), __FILE__, __LINE__)
