module SharedLogging

using Logging
using Dates

# Set up logging constants
const LOG_LEVEL = Logging.Info

"""
    setup_logging(component_name::String)

Initialize logging for a specific component with proper file handling.
"""
function setup_logging(component_name::String)
    try
        log_dir = joinpath(@__DIR__, "logs")
        mkpath(log_dir)  # Create logs directory if it doesn't exist
        log_file = joinpath(log_dir, "$(component_name)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log")
        logger = SimpleLogger(open(log_file, "a"))
        global_logger(logger)
        @info "Starting $component_name" timestamp=now() logfile=log_file
        return logger
    catch e
        @error "Failed to setup logging" component=component_name error=e
        rethrow(e)
    end
end

export LOG_LEVEL, setup_logging

end # module