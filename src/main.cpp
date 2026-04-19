#include <iostream>
#include <string>

#include <spdlog/spdlog.h>

#include "core/config.h"
#include "core/logger.h"
#include "core/pipeline.h"

int main(int argc, char* argv[]) {
    std::string config_path = "config/pipeline.yaml";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: drone_tracker [--config <path>]\n"
                      << "\nKeyboard controls:\n"
                      << "  Q/ESC  Quit\n"
                      << "  T      Cycle target selection mode\n"
                      << "  1-9    Select track by ID\n"
                      << "  R      Toggle recording\n"
                      << "  D      Toggle debug overlay\n";
            return 0;
        }
    }

    try {
        auto config = drone_tracker::Config::load(config_path);
        drone_tracker::init_logger(config.logging);

        spdlog::info("Drone Tracker v0.1.0");
        spdlog::info("Config: {}", config_path);

        drone_tracker::Pipeline pipeline(config);
        pipeline.start();

    } catch (const std::exception& e) {
        spdlog::error("Fatal: {}", e.what());
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
