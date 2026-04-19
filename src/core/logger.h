#pragma once

#include <string>

namespace drone_tracker {

struct LoggingConfig;

void init_logger(const LoggingConfig& config);

}  // namespace drone_tracker
