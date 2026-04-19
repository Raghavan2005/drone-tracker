#include "core/logger.h"
#include "core/config.h"

#include <filesystem>
#include <vector>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace drone_tracker {

void init_logger(const LoggingConfig& config) {
    std::vector<spdlog::sink_ptr> sinks;

    if (config.console) {
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    }

    if (!config.file.empty()) {
        std::filesystem::create_directories(std::filesystem::path(config.file).parent_path());
        sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.file, true));
    }

    auto logger = std::make_shared<spdlog::logger>("drone_tracker", sinks.begin(), sinks.end());

    if (config.level == "trace") logger->set_level(spdlog::level::trace);
    else if (config.level == "debug") logger->set_level(spdlog::level::debug);
    else if (config.level == "info") logger->set_level(spdlog::level::info);
    else if (config.level == "warn") logger->set_level(spdlog::level::warn);
    else if (config.level == "error") logger->set_level(spdlog::level::err);
    else logger->set_level(spdlog::level::info);

    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");
    spdlog::set_default_logger(logger);
    spdlog::info("Logger initialized (level={})", config.level);
}

}  // namespace drone_tracker
