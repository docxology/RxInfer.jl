println("🚀 Starting setup for RxInfer MountainCar example...")

import Pkg

# Create a new project environment in current directory
println("\n📦 Creating new project environment...")
Pkg.activate(@__DIR__)

# Remove existing RxInfer if present
println("\n🔄 Checking for existing RxInfer package...")
try
    Pkg.rm("RxInfer")
    println("✅ Removed existing RxInfer package")
catch e
    println("ℹ️  No existing RxInfer package found")
end

# Add RxInfer from local path
println("\n📥 Adding local RxInfer package...")
rxinfer_path = joinpath(dirname(@__DIR__), "..")  # Go up two directories to RxInfer.jl root
try
    Pkg.develop(path=rxinfer_path)
    println("✅ Successfully added local RxInfer package")
catch e
    println("❌ Failed to add local RxInfer package:")
    println("   $e")
    exit(1)
end

# Add other required packages
println("\n📥 Installing additional packages:")
deps = [
    "Plots",
    "Printf",
    "Statistics",
    "Distributions",
    "HypergeometricFunctions"
]

for package in deps
    print("   Installing $package...")
    try
        Pkg.add(package)
        println(" ✅")
    catch e
        println(" ❌")
        println("   Error installing $package: $e")
        exit(1)
    end
end

# Resolve dependencies
println("\n🔄 Resolving dependencies...")
Pkg.resolve()
Pkg.instantiate()

# Verify installation
println("\n🔍 Verifying installations...")
try
    # Test loading each package
    print("   Loading RxInfer...")
    eval(Meta.parse("using RxInfer"))
    println(" ✅")
    
    for package in deps
        print("   Loading $package...")
        eval(Meta.parse("using $package"))
        println(" ✅")
    end
    
    println("\n✨ Setup completed successfully!")
    println("🎮 You can now run the MountainCar example by executing:")
    println("   julia MountainCar.jl")
catch e
    println("\n❌ Setup failed:")
    println("   $e")
    exit(1)
end
