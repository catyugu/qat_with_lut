#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <string>
#include <map>

// 一个辅助类，利用其构造和析构函数来自动计时一个作用域
// An auxiliary class that uses its constructor and destructor to automatically time a scope
class ScopedTimer {
public:
    ScopedTimer(const std::string& name);
    ~ScopedTimer();
private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

// 全局单例分析器，用于收集和报告数据
// Global singleton profiler for collecting and reporting data
class Profiler {
public:
    static Profiler& getInstance();
    void addRecord(const std::string& name, long long duration_us);
    void report();
    void reset();

private:
    Profiler() = default; // Private constructor for singleton
    struct ProfileData {
        long long total_us = 0;
        long long count = 0;
    };
    std::map<std::string, ProfileData> records_;
};

// 宏定义，让计时代码更简洁
// Macro definition to make timing code more concise
#define PROFILE_SCOPE(name) ScopedTimer timer(name)

#endif // PROFILER_H