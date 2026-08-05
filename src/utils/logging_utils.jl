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

Sets a GLOBAL logging floor via `Logging.disable_logging`, independent of
whatever logger object is currently active (VSCodeLogger, ConsoleLogger,
etc.) and independent of which task/thread emits the message. This is
checked before any per-logger `shouldlog` dispatch, so it reliably blocks
noisy library `@debug` output (e.g. GPUCompiler inference remarks during
CUDA kernel compilation) that per-module logger filtering failed to catch.

Trade-off: this is an all-or-nothing floor — when `level == INFO` (default),
ALL `@debug` calls process-wide are suppressed, including Thesis's own. Set
`LOG_LEVEL=DEBUG` to allow ALL @debug through (yours AND every library's).
There is no reliable way to show only your own @debug while blocking every
library's, given how GPUCompiler's remarks are emitted — attempts at
per-module logger filtering (EarlyFilteredLogger / custom AbstractLogger)
did not reliably intercept them in this environment.
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