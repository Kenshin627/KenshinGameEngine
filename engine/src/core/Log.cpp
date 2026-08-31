#include "kenshin/core/Log.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <filesystem>

KENSHIN_NS_BEGIN

namespace log
{
    namespace
    {
        std::shared_ptr<spdlog::logger> s_logger;

        constexpr const char* kPattern = "[%H:%M:%S.%e] [%^%l%$] %v";
    } // namespace

    void init(const char* logDir)
    {
        if (s_logger)
            return; // idempotent

        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(logDir, ec); // already-exists is not an error

        auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console->set_pattern(kPattern);

        auto file = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            (fs::path(logDir) / "kenshin.log").string(), true /*truncate on start*/);
        file->set_pattern(kPattern);

        s_logger = std::make_shared<spdlog::logger>(
            "kenshin", spdlog::sinks_init_list{console, file});
        s_logger->set_level(spdlog::level::trace);
        s_logger->flush_on(spdlog::level::warn);

        spdlog::set_default_logger(s_logger);
    }

    void shutdown()
    {
        if (s_logger)
        {
            s_logger->flush();
            s_logger.reset();
        }
    }

    spdlog::logger* logger()
    {
        return s_logger ? s_logger.get() : spdlog::default_logger_raw();
    }

} // namespace log

KENSHIN_NS_END
