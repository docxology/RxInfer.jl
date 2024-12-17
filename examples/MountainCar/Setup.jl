#!/usr/bin/env julia

println("🚀 Starting setup for RxInfer MountainCar example...")

import Pkg
using Pkg: PackageSpec

# Define paths
const RXINFER_PATH = joinpath(dirname(@__DIR__), "..")  # Go up two directories to RxInfer.jl root
const CURRENT_DIR = @__DIR__

# Function to safely add package with version
function safe_add_package(pkg_name, version=nothing)
    try
        if pkg_name in keys(Pkg.project().dependencies)
            println("📦 $pkg_name is already installed")
            return true
        end
        println("📥 Installing $pkg_name...")
        if isnothing(version)
            Pkg.add(pkg_name)
        else
            Pkg.add(PackageSpec(name=pkg_name, version=version))
        end
        println("✅ Successfully installed $pkg_name")
        return true
    catch e
        println("❌ Failed to install $pkg_name:")
        println("   $e")
        return false
    end
end

# Create and activate project environment
println("\n📦 Creating project environment...")
try
    Pkg.activate(CURRENT_DIR)
    println("✅ Activated project at $(Pkg.project().path)")
catch e
    println("❌ Failed to activate project:")
    println("   $e")
    exit(1)
end

# Required packages with specific versions
println("\n📦 Installing required packages...")
required_packages = [
    ("Plots", "1.38"),
    ("DataFrames", "1.7"),
    ("CSV", "0.10"),
    ("Distributions", "0.25"),
    ("LinearAlgebra", nothing),
    ("Statistics", nothing),
    ("Random", nothing),
    ("Logging", nothing),
    ("LoggingExtras", "1"),
    ("TOML", "1"),
    ("ProgressMeter", "1"),
    ("Dates", nothing),
    ("Printf", nothing),
    ("HypergeometricFunctions", "0.3"),
    ("SharedArrays", nothing),
    ("Distributed", nothing),
    ("Measures", "0.3"),
    ("Colors", "0.12"),
    ("RecipesBase", "1"),
    ("Parameters", "0.12"),
    ("StatsBase", "0.34")
]

# Install packages
failed_packages = String[]
for (pkg, version) in required_packages
    if !safe_add_package(pkg, version)
        push!(failed_packages, pkg)
    end
end

# Add ReactiveMP and RxInfer in the correct order
println("\n📥 Installing ReactiveMP and RxInfer packages...")
try
    # First install ReactiveMP with specific version
    Pkg.add(PackageSpec(name="ReactiveMP", version="4.4.5"))
    println("✅ Successfully installed ReactiveMP")
    
    # First activate and build RxInfer
    Pkg.activate(RXINFER_PATH)
    Pkg.instantiate()
    Pkg.build()
    println("✅ Built RxInfer in root project")
    
    # Switch back to MountainCar environment
    Pkg.activate(CURRENT_DIR)
    
    # Remove any existing RxInfer
    if "RxInfer" in keys(Pkg.project().dependencies)
        Pkg.rm("RxInfer")
    end
    
    # Develop RxInfer from local path
    Pkg.develop(PackageSpec(path=RXINFER_PATH))
    println("✅ Successfully added local RxInfer package")
catch e
    println("❌ Failed to install packages:")
    println("   $e")
    exit(1)
end

# Resolve and instantiate dependencies
println("\n🔄 Resolving dependencies...")
try
    Pkg.resolve()
    Pkg.instantiate()
    
    # Verify packages are installed
    @eval using ReactiveMP
    @eval using RxInfer
    println("✅ Dependencies resolved successfully")
catch e
    println("❌ Failed to resolve dependencies:")
    println("   $e")
    exit(1)
end

# Create necessary directories
println("\n📁 Creating necessary directories...")
try
    # Main output directory
    mkpath(joinpath(CURRENT_DIR, "Outputs"))
    println("✅ Created Outputs directory")
    
    # Meta-analysis output directory
    mkpath(joinpath(CURRENT_DIR, "MetaAnalysis_Outputs"))
    println("✅ Created MetaAnalysis_Outputs directory")
catch e
    println("❌ Failed to create directories:")
    println("   $e")
end

# Print setup summary
println("\n📋 Setup Summary:")
if isempty(failed_packages)
    println("✅ All packages installed successfully")
else
    println("⚠️  Failed to install packages:")
    for pkg in failed_packages
        println("   - $pkg")
    end
end

# Print usage instructions
println("\n🎮 Usage Instructions:")
println("1. For single simulation:")
println("   julia --project=. MountainCar.jl")
println("\n2. For parameter sweep (choose one):")
println("   Multi-threaded (recommended):")
println("      JULIA_NUM_THREADS=4 julia --project=. MetaAnalysis_MountainCar.jl")

# Print environment info
println("\n💻 Environment Information:")
println("Julia Version: $(VERSION)")
println("Number of Threads: $(Threads.nthreads())")
println("Project Path: $(Pkg.project().path)")
println("RxInfer Path: $(pathof(RxInfer))")

if !isempty(failed_packages)
    exit(1)
end