#include "profiler.h"
#include <iostream>
#include <iomanip> // For std::setw, std::left, etc.

Profiler& Profiler::getInstance() {
    static Profiler instance;
    return instance;
}

void Profiler::addRecord(const std::string& name, long long duration_us) {
    // This function is thread-safe for different names, but not for the same name.
    // For full thread-safety with OpenMP, a mutex would be needed here.
    // For now, it's sufficient for single-threaded analysis.
    records_[name].total_us += duration_us;
    records_[name].count++;
}

void Profiler::report() {
    std::cout << "\n--- Profiling Report ---\n";
    std::cout << std::left << std::setw(25) << "Function"
              << std::setw(20) << "Total Time (ms)"
              << std::setw(15) << "Calls"
              << std::setw(20) << "Avg Time (us)" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& pair : records_) {
        double total_ms = pair.second.total_us / 1000.0;
        double avg_us = static_cast<double>(pair.second.total_us) / pair.second.count;
        std::cout << std::left << std::setw(25) << pair.first
                  << std::setw(20) << std::fixed << std::setprecision(3) << total_ms
                  << std::setw(15) << pair.second.count
                  << std::setw(20) << std::fixed << std::setprecision(3) << avg_us << "\n";
    }
     std::cout << "--------------------------------------------------------------------------------\n";
}

void Profiler::reset() {
    records_.clear();
}


// --- ScopedTimer Implementation ---
ScopedTimer::ScopedTimer(const std::string& name)
    : name_(name), start_time_(std::chrono::high_resolution_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count();
    Profiler::getInstance().addRecord(name_, duration);
}