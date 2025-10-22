using LinearAlgebra
using Printf

#------------------------------------------------------------#
# Helper functions
function sep(title)
    println("\n", "="^60)
    println(title)
    println("="^60, "\n")
end

function pause()
    print("\nPress ENTER to continue..."); readline()
end

#------------------------------------------------------------#
# Element-wise vs Matrix multiplication (2D)
function demo_elementwise_vs_matrix()
    sep("Element-wise (.*) vs Matrix (*) Multiplication (2D)")

    A = rand(3,3)
    B = rand(3,3)
    println("A (3×3):"); show(stdout, "text/plain", A); println("\n")
    println("B (3×3):"); show(stdout, "text/plain", B); println("\n")

    E = A .* B
    M = A * B

    println("A .* B  (element-wise):"); show(stdout, "text/plain", E); println("\n")
    println("A * B   (matrix product):"); show(stdout, "text/plain", M); println("\n")

    println("Notes:")
    println("'.*' multiplies each corresponding entry (Hadamard product).")
    println("'*' does row-by-column dot products; result shape is (m×p) for (m×n)*(n×p).")
    pause()
end

#------------------------------------------------------------#
# Whole-tensor dot equivalence (3D)
function demo_tensor_dot_equivalents()
    sep("Whole-tensor dot on 3D arrays: sum(B .* C) ≡ dot(vec(B), vec(C))")

    B = rand(2,3,4)
    C = rand(2,3,4)
    d2 = sum(B .* C)
    d3 = dot(vec(B), vec(C))

    @printf "sum(B .* C)        = %.6f\n" d2
    @printf "dot(vec(B),vec(C)) = %.6f\n" d3
    println("Equal? ", d2 == d3, " (isapprox: ", isapprox(d2, d3), ")")
    println("\nReminder: '*' (matrix multiply) is undefined for raw 3D arrays.")
    pause()
end

#------------------------------------------------------------#
# Per-slice (batched) matrix multiplication for 3D arrays
function demo_batched_slice_matmul()
    sep("Per-slice (batched) matrix multiplication for 3D arrays")

    B = rand(2,3,4)
    C = rand(3,2,4)

    result_slices = [B[:,:,i] * C[:,:,i] for i in 1:size(B,3)]
    R = cat(result_slices..., dims=3)

    println("B size: ", size(B), "   C size: ", size(C))
    println("Per-slice result size (stacked): ", size(R))
    println("\nFirst slice result R[:,:,1]:")
    show(stdout, "text/plain", R[:,:,1]); println()
    println("\nTip: Use this pattern when you mean 'batched' matmul.")
    pause()
end

#------------------------------------------------------------#
# Broadcasting & Constructors
function demo_broadcasting_and_constructors()
    sep("Broadcasting & Common Constructors (quick demo)")

    x  = Float32.(0:11)
    y  = Float32.(0:2:12)
    x1 = collect(Float32.(0:11))
    x2 = collect(Float32.(0:2:12))
    X3 = Float32.([1 2 3; 4 5 6; 7 8 9])
    sample = 1:9
    X71 = reshape(Float64.(collect(sample)), 3, 3)
    X72 = reshape([Float64(i) for i in sample], 3, 3)

    println("x  = Float32.(0:11):"); show(stdout, "text/plain", x); println("\n")
    println("y  = Float32.(0:2:12):"); show(stdout, "text/plain", y); println("\n")
    println("X3 = Float32.([1 2 3; 4 5 6; 7 8 9]):"); show(stdout, "text/plain", X3); println("\n")
    println("X71 reshape from range -> 3×3 Float64:"); show(stdout, "text/plain", X71); println("\n")
    println("X72 same via comprehension -> 3×3 Float64:"); show(stdout, "text/plain", X72); println("\n")
    println("Equivalence X71 == X72? ", X71 == X72)
    pause()
end

#------------------------------------------------------------#
# Compare full dot vs per-slice multiplication
function demo_full_vs_slice_dot()
    sep("Comparing full-tensor dot vs per-slice matrix multiplication")

    B = rand(2,3,4)
    C = rand(3,2,4)

    d_full = sum(B .* permutedims(C, (2,1,3))[1:2,1:3,1:4])
    println("Full dot-style sum(B .* C) across all dims (approx):")
    @printf "  → Scalar value: %.6f\n\n" d_full

    result_slices = [B[:,:,i] * C[:,:,i] for i in 1:size(B,3)]
    R = cat(result_slices..., dims=3)

    println("Per-slice results:")
    println("  R size = ", size(R))
    println("  Example slice R[:,:,1]:")
    show(stdout, "text/plain", R[:,:,1]); println("\n")

    println("Conceptual difference:")
    println("Full dot: sums *all* elementwise products into one scalar (like flattening both).")
    println("Per-slice: performs 4 independent (2×3)*(3×2) matrix multiplications → (2×2×4).")
    println("So: full dot measures *overall similarity*, per-slice gives *structured batch results*.")
    pause()
end

#------------------------------------------------------------#
# Dot product using transpose (x' * y) for 1D and 2D arrays
function demo_transpose_dot()
    sep("Dot Product using Transpose (x' * y) in Julia")

    # --- 1D example
    x = [1, 2, 3]
    y = [4, 5, 6]
    println("1D vectors:")
    println("x = ", x)
    println("y = ", y)
    println("\nCompute:")
    println("x' * y = ", x' * y)
    println("dot(x, y) = ", dot(x, y))
    println("\nExplanation:")
    println("For 1D vectors, x' (transpose) turns the column vector [1;2;3] into a row [1 2 3].")
    println("Then row × column = scalar: 1×4 + 2×5 + 3×6 = 32.")
    println("So x' * y is equivalent to dot(x, y).")
    println("x' * y returns a 1×1 Matrix; dot(x,y) returns a scalar Float64.\n")

    # --- 2D example
    A = [1 2; 3 4; 5 6]  # 3×2
    B = [2 0; 1 3; 4 5]  # 3×2
    println("2D matrices:")
    println("A (3×2):"); show(stdout, "text/plain", A); println("\n")
    println("B (3×2):"); show(stdout, "text/plain", B); println("\n")

    println("Now compute A' * B:")
    println("A' * B =")
    show(stdout, "text/plain", A' * B); println("\n")

    println("Explanation:")
    println("A is 3×2, so A' (transpose) is 2×3.")
    println("A' * B multiplies (2×3) × (3×2) → (2×2).")
    println("Each entry of A' * B is the dot product between one column of A and one column of B.")
    println("This generalizes the vector dot product to matrices — it gives all pairwise dot products between columns of A and B.")
    pause()
end

#------------------------------------------------------------#
# Main Menu Loop
function menu()
    while true
        println("\n===== Tensor & Matrix Ops Menu =====")
        println("1) Element-wise (.*) vs Matrix (*) multiplication (2D)")
        println("2) 3D tensors: sum(B .* C) vs dot(vec(B), vec(C))")
        println("3) Per-slice (batched) matrix multiplication (3D)")
        println("4) Broadcasting & common constructors demo")
        println("5) Compare full-tensor dot vs per-slice matrix multiplication")
        println("6) Dot product using transpose (x' * y) — 1D & 2D examples")
        println("0) Exit")
        print("Choose an option: ")
        choice = tryparse(Int, chomp(readline()))
        choice === nothing && continue

        if choice == 1
            demo_elementwise_vs_matrix()
        elseif choice == 2
            demo_tensor_dot_equivalents()
        elseif choice == 3
            demo_batched_slice_matmul()
        elseif choice == 4
            demo_broadcasting_and_constructors()
        elseif choice == 5
            demo_full_vs_slice_dot()
        elseif choice == 6
            demo_transpose_dot()
        elseif choice == 0
            println("Bye!"); break
        else
            println("Invalid choice.")
        end
    end
end

#------------------------------------------------------------#
# Run the menu if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    menu()
end
