# src/utils/logging_utils.jl
#
# Global logging configuration utilities to control console verbosity and debug output across the codebase.

using Logging

export set_log_level!, get_log_level

const LOG_LEVEL_MAP = Dict(
    "DEBUG" => Logging.Debug,
    "INFO"  => Logging.Info,
    "WARN"  => Logging.Warn,
    "ERROR" => Logging.Error,
)

"""
    set_log_level!(level=get(ENV, "LOG_LEVEL", "INFO"))

Sets a global loglevel. This was implemented to make the @debug annotation possible. Note that many used libaries also use @debug which makes the debug-log amount very high.

Parameters:
    - 'level': The loglevel that should be logged. Is taken from the provided .env file.
"""
function set_log_level!(level::Union{String,Logging.LogLevel}=get(ENV, "LOG_LEVEL", "INFO"))
    lvl = level isa Logging.LogLevel ? level : get(LOG_LEVEL_MAP, uppercase(level), Logging.Info)

    # Disable all logging strictly below `lvl`. E.g. lvl=Info disables Debug.
    floor = lvl == Logging.Debug ? Logging.BelowMinLevel : (lvl - 1)
    Logging.disable_logging(floor)

    Logging.global_logger(Logging.ConsoleLogger(stderr, lvl))
    @info "Log level set to $lvl (global floor: $floor)"
end

get_log_level() = Logging.min_enabled_level(Logging.global_logger())