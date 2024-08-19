include("./BackscatterSimulation.jl")

energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_data_bins()

# Create empty distribution
input_distribution = zeros(energy_nbins, pa_nbins)

# Find region where our beam is
energy_indices = 1000 .≤ energy_bin_means .≤ 1100
pitch_angle_indices = 45 .≤ pa_bin_means .≤ 50

# Create beam
input_distribution[energy_indices, pitch_angle_indices] .= 10

plot_distribution(input_distribution)

distributions = backscatter_simulation(input_distribution, n_bounces = 1)

plot_distribution(distributions[2,:,:])