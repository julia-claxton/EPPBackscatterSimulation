using Statistics
using LinearAlgebra

# This is a file of general-purpose functions that I find useful, compiled in one
# script so I can use them in all my scripts.

# Written by Julia Claxton. Contact: julia.claxton@colorado.edu
# Released under MIT License (see <project top level>/LICENSE.txt for full license)

function print_progress_bar(fraction; bar_length = 20, overwrite = true)
# Prints a progress bar to terminal filled to user-specified percentage.
    character_length = 1/bar_length
    number_of_filled_characters = Int(floor(fraction/character_length))
    if overwrite == true; print("\r"); end
    print(repeat("█", number_of_filled_characters))
    print(repeat("░", bar_length - number_of_filled_characters))
    print(" [$(round(fraction*100, digits = 1))%]")
end

function edges_to_means(edges)
# Calculates means of bins given their edges.
    return [mean([edges[i], edges[i+1]]) for i = 1:length(edges)-1] 
end

function exact_2dhistogram(x, y, x_bin_edges, y_bin_edges; weights = ones(length(x)))
# Bin 2D data into a histogram with bin edges defined by user.
    @assert length(x) == length(y) == length(weights) "x, y, and weight vectors must be same length"

    x_nbins = length(x_bin_edges) - 1
    y_nbins = length(y_bin_edges) - 1
    
    result = zeros(x_nbins, y_nbins)
    
    for x_idx = 1:x_nbins
        x_slice = x_bin_edges[x_idx] .<= x .< x_bin_edges[x_idx+1]
        for y_idx = 1:y_nbins
            y_slice = y_bin_edges[y_idx] .<= y .< y_bin_edges[y_idx+1]
            slice = x_slice .&& y_slice
            result[x_idx, y_idx] = sum(weights[slice])
        end
    end
    return result
end

function exact_1dhistogram(data, bin_edges; weights = ones(length(data)))
    # Bin 1D data into a histogram with bin edges defined by user.
    @assert length(data) == length(weights) "Data and weight vectors must be same length"
    nbins = length(bin_edges) - 1
    
    result = zeros(nbins)
    for idx = 1:nbins
        slice = bin_edges[idx] .<= data .< bin_edges[idx+1]
        result[idx] = sum(weights[slice])
    end
    return result
end
    
function meshgrid(x, y)
    grid = [(x[i], y[j]) for i in eachindex(x), j in eachindex(y)]
    return reshape(grid, :) # Vectorize
end