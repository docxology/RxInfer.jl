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

# Function to verify package can be loaded
function verify_package(pkg_name)
    try
        @eval using $(Symbol(pkg_name))
        println("✅ Successfully loaded $pkg_name")
        return true
    catch e
        println("❌ Failed to load $pkg_name:")
        println("   $e")
        return false
    end
end

function main()
    # Create and activate project environment
    println("\n📦 Creating project environment...")
    try
        Pkg.activate(CURRENT_DIR)
        println("✅ Activated project at $(Pkg.project().path)")
    catch e
        println("❌ Failed to activate project:")
        println("   $e")
        return false
    end

    # First, install and build RxInfer
    println("\n📦 Setting up RxInfer...")

    # First activate and build RxInfer in its own environment
    println("🔨 Building RxInfer in its own environment...")
    try
        Pkg.activate(RXINFER_PATH)
        Pkg.instantiate()
        Pkg.build()
        println("✅ Built RxInfer in its own environment")
    catch e
        println("❌ Failed to build RxInfer in its environment:")
        println("   $e")
        return false
    end

    # Switch back to MountainCar environment
    Pkg.activate(CURRENT_DIR)

    # Remove any existing RxInfer
    if "RxInfer" in keys(Pkg.project().dependencies)
        println("🗑️  Removing existing RxInfer...")
        Pkg.rm("RxInfer")
    end

    # Add RxInfer from local path
    println("📥 Adding RxInfer from local path...")
    try
        Pkg.develop(PackageSpec(path=RXINFER_PATH))
        println("✅ Successfully added RxInfer from local path")
    catch e
        println("❌ Failed to add RxInfer:")
        println("   $e")
        return false
    end

    # Required packages with specific versions
    println("\n📦 Installing required packages...")
    required_packages = [
        ("Plots", "1.40"),
        ("DataFrames", "1.7"),
        ("CSV", "0.10"),
        ("UnicodePlots", "3.6"),
        ("ProgressMeter", "1.10"),
        ("TOML", "1.0"),
        ("HypergeometricFunctions", "0.3"),
        ("LinearAlgebra", nothing),  # Part of stdlib
        ("Statistics", nothing),     # Part of stdlib
        ("Printf", nothing),         # Part of stdlib
        ("Dates", nothing)          # Part of stdlib
    ]

    installation_success = true
    for (pkg, version) in required_packages
        if !safe_add_package(pkg, version)
            installation_success = false
        end
    end

    if !installation_success
        println("\n❌ Some packages failed to install.")
        return false
    end

    # Create necessary directories
    println("\n📁 Creating necessary directories...")
    dir_creation_success = true
    for dir in ["results", "logs", "Outputs"]
        dir_path = joinpath(CURRENT_DIR, dir)
        try
            mkpath(dir_path)
            println("✅ Created directory: $dir")
        catch e
            println("❌ Failed to create directory $dir:")
            println("   $e")
            dir_creation_success = false
        end
    end

    if !dir_creation_success
        println("\n❌ Failed to create some directories.")
        return false
    end

    # Verify installations
    println("\n🔍 Verifying package installations...")
    verification_success = true

    # First verify RxInfer
    println("\n📋 Verifying RxInfer installation...")
    try
        @eval using RxInfer
        # Try to create a simple model to verify functionality
        @eval begin
            @model function test_model()
                x ~ Normal(0.0, 1.0)
                y ~ Normal(x, 1.0)
                return y
            end
        end
        println("✅ Successfully verified RxInfer functionality")
    catch e
        println("❌ Failed to verify RxInfer:")
        println("   $e")
        verification_success = false
    end

    # Verify other key packages
    key_packages = ["Plots", "DataFrames", "CSV", "UnicodePlots", "ProgressMeter", "TOML"]
    for pkg in key_packages
        if !verify_package(pkg)
            verification_success = false
        end
    end

    if !verification_success
        println("\n❌ Some package verifications failed.")
        return false
    end

    # Print environment information
    println("\n💻 Environment Information:")
    println("Julia Version: $(VERSION)")
    println("Project Path: $(Pkg.project().path)")
    println("RxInfer Path: $RXINFER_PATH")

    # Final status
    println("\n✅ Setup completed successfully!")
    println("You can now run the MountainCar analyses with:")
    println("   julia --project=. MetaAnalysis_MountainCar.jl")
    println("   julia --project=. MountainCar_Standalone.jl")

    
    return true
end

# Run main function and exit with appropriate status
if !main()
    exit(1)
end