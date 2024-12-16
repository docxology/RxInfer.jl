println("🚀 Starting setup for RxInfer Biofirm example...")

import Pkg

# Create a new project environment in current directory
println("\n📦 Creating new project environment...")
Pkg.activate(".")

# Add required packages with status logging
required_packages = [
    "RxInfer",
    "Plots",
    "HypergeometricFunctions"
]

println("\n📥 Installing required packages:")
for package in required_packages
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

# Verify installation
println("\n🔍 Verifying installations...")
try
    # Test loading each package
    for package in required_packages
        print("   Loading $package...")
        eval(Meta.parse("using $package"))
        println(" ✅")
    end
    
    println("\n✨ Setup completed successfully!")
    println("🎮 You can now run the Biofirm example by executing:")
    println("   julia Biofirm.jl")
catch e
    println("\n❌ Setup failed:")
    println("   $e")
    exit(1)
end
