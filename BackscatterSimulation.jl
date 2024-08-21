# Housekeeping
const BackscatterSimulation_TOP_LEVEL = @__DIR__
@assert endswith(BackscatterSimulation_TOP_LEVEL, "EPPBackscatterSimulation")

# Library includes
using Statistics
using LinearAlgebra
using NPZ
using Plots
using Plots.PlotMeasures
using DelimitedFiles
using Glob

# ---------------- Backscatter Simulation Functions ----------------
function backscatter_simulation(input_distribution, n_bounces, show_progress = true)
    energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_data_bins()

    distributions = zeros(n_bounces + 1, energy_nbins, pa_nbins)
    distributions[1,:,:] = input_distribution
    for i = 2:(n_bounces+1)
        if show_progress == true; println("Bounce $(i-1)/$(n_bounces)"); end
        input_distribution = distributions[i-1,:,:]

        if i > 2 # Only do this for distributions after the first
            reverse!(input_distribution, dims = 2) # Move the distribution from the antiloss cone to the loss cone
        end

        distributions[i,:,:] = simulate_backscatter(input_distribution, show_progress = show_progress)
    end
    # Now we need to reverse the even bounces to get them in the correct cone from the persepctive of a stationary observer
    # Since we start at the 0th bounce, the 2nd, 4th, etc, bounces will be indexes 3, 5, 7, etc.
    distributions_to_reverse = 3:2:(n_bounces+1)
    for i in distributions_to_reverse
        distributions[i, :, :] = reverse(distributions[i, :, :], dims = 2)
    end
    if show_progress == true; println(); end
    return distributions
end

function simulate_backscatter(input_distribution; show_progress = true)
    energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_data_bins()

    # Create backscatter distribution
    output_distribution = zeros(energy_nbins, pa_nbins)
    pixels_to_simulate = findall(input_distribution .≠ 0) # Don't simulate pixels with zero input flux
    for i = 1:length(pixels_to_simulate)
        # Print progress to terminal
        # But don't do it on every iteration because that's a big slowdown
        if show_progress && ((i % 50 == 0) || (i == length(pixels_to_simulate)))
            _print_progress_bar(i/length(pixels_to_simulate))
        end

        idx = pixels_to_simulate[i]
        e_idx = idx[1]
        α_idx = idx[2]

        pixel_e = energy_bin_means[e_idx]
        pixel_α = pa_bin_means[α_idx]
        if pixel_α > SIMULATION_α_MAX; continue; end # Prebaked distributions only go to 70º with bin width = 5º

        backscatter = _get_prebaked_backscatter(pixel_e, pixel_α) * input_distribution[idx]
        @assert size(backscatter) == size(output_distribution) "Prebaked backscatter grid does not match simulation grid"
        output_distribution .+= backscatter
    end
    if show_progress == true; println(); end
    return output_distribution
end

function _get_prebaked_backscatter(input_energy, input_pa)
    # Given a pitch angle and energy of an input electron beam, retrieves nearest precalculated backscatter distribution. Code 
    # adapted from https://github.com/GrantBerland/G4EPP/blob/main/G4EPP/examples/invert_ELFIN_measurements.ipynb and ported
    # to Julia by me.
    energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_data_bins()
    
    # Kick out bad inputs
    if (length(input_energy) != 1) || (length(input_pa) != 1)
        error("Input energy and pitch angle must have length 1")
    end
    if (5 <= input_energy <= 15e3) == false # Data is provided from 10keV to 10e3keV, adding some margin
        return zeros(energy_nbins, pa_nbins)
    end
    if (0 <= input_pa <= SIMULATION_α_MAX) == false
        return zeros(energy_nbins, pa_nbins)
    end

    # Find precalculated beam energies and pitch angles
    backscatter_directory = "$(BackscatterSimulation_TOP_LEVEL)/data/binned_backscatter_distributions"
    backscatter_filepaths = glob("*deg.npz", backscatter_directory)
    backscatter_filenames = replace.(backscatter_filepaths, "$(backscatter_directory)/" => "")

    # RegEx matching to get energies
    backscatter_energies = match.(r"(.*?)keV_", backscatter_filenames)
    backscatter_energies = [backscatter_energies[i].captures[1] for i = eachindex(backscatter_energies)]
    backscatter_energies = parse.(Int64, backscatter_energies)

    # RegEx matching to get pitch angles
    backscatter_pitch_angles = match.(r"keV_(.*?)deg.npz", backscatter_filenames)
    backscatter_pitch_angles = [backscatter_pitch_angles[i].captures[1] for i = eachindex(backscatter_pitch_angles)]
    backscatter_pitch_angles = parse.(Int64, backscatter_pitch_angles)

    # Match input to nearest precalculated beam
    energy_differences = abs.(backscatter_energies .- input_energy)
    mindistance, _ = findmin(energy_differences)
    energy_idxs = findall(energy_differences .== mindistance)

    pa_differences = abs.(backscatter_pitch_angles .- input_pa)
    mindistance, _ = findmin(pa_differences)
    pa_idxs = findall(pa_differences .== mindistance)

    nearest_beam_idx = intersect(energy_idxs, pa_idxs)
    nearest_beam_idx = nearest_beam_idx[1] # Extract index from 1-element vector. If we are exactly halfway between two or more beams, this will round down

    filename = backscatter_filenames[nearest_beam_idx]

    # Return prebaked backscatter distribution
    data = npzread("$(backscatter_directory)/$(filename)")
    return data["backscatter_distribution"]
end

function get_data_bins()
    data_bins = npzread("$(BackscatterSimulation_TOP_LEVEL)/data/binned_backscatter_distributions/data_bins.npz")

    energy_nbins = data_bins["energy_nbins"]
    energy_bin_edges = data_bins["energy_bin_edges"]
    energy_bin_means = data_bins["energy_bin_means"]

    pa_nbins = data_bins["pa_nbins"]
    pa_bin_edges = data_bins["pa_bin_edges"]
    pa_bin_means = data_bins["pa_bin_means"]

    SIMULATION_α_MAX = 72.5

    return energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX
end

# ---------------- Backscatter Binning Functions ----------------
function bin_backscatter_data(energy_nbins, pa_nbins)
    # Calculate bins
    energy_bin_edges = 10 .^ LinRange(1, 4, energy_nbins + 1)
    pa_bin_edges = LinRange(0, 180, pa_nbins + 1)

    # Save bins
    npzwrite("$(BackscatterSimulation_TOP_LEVEL)/data/binned_backscatter_distributions/data_bins.npz",
        energy_nbins = energy_nbins,
        energy_bin_edges = energy_bin_edges,
        energy_bin_means = _bin_edges_to_means(energy_bin_edges),

        pa_nbins = pa_nbins,
        pa_bin_edges = pa_bin_edges,
        pa_bin_means = _bin_edges_to_means(pa_bin_edges)
    )

    # Find backscatter data
    backscatter_data_directory = "$(BackscatterSimulation_TOP_LEVEL)/data/raw_backscatter"
    backscatter_filepaths = glob("*.csv", backscatter_data_directory)
    backscatter_filenames = replace.(backscatter_filepaths, "$(backscatter_data_directory)/" => "")

    # Make sure we have a destination path for the binned distributions
    if isdir("$(BackscatterSimulation_TOP_LEVEL)/data/binned_backscatter_distributions") == false
        mkdir("$(BackscatterSimulation_TOP_LEVEL)/data/binned_backscatter_distributions")
    end

    # Start binning data
    println("Binning backscatter distributions...")
    println("\tEnergy Bins = $(energy_nbins)")
    println("\tPitch Angle Bins = $(pa_nbins)")

    for i = 1:length(backscatter_filenames)
        _print_progress_bar(i/length(backscatter_filenames))
        _prebake_backscatter_file(backscatter_filenames[i], backscatter_data_directory)
    end
    println("\n")
end

function _prebake_backscatter_file(filename, backscatter_data_directory)
    # Get beam parameters from filename
    energy = match.(r"bs_spectra(.*?)keV", filename)[1]
    energy = parse(Int64, energy)

    pitch_angle = match.(r"PAD(.*?).csv", filename)[1]
    pitch_angle = parse(Int64, pitch_angle)

    backscatter_distribution = _get_single_beam_backscatter(filename, backscatter_data_directory)

    npzwrite("$(BackscatterSimulation_TOP_LEVEL)/data/binned_backscatter_distributions/$(energy)keV_$(pitch_angle)deg.npz",
        energy_bin_edges = energy_bin_edges,
        pitch_angle_bin_edges = pa_bin_edges,
        backscatter_distribution = backscatter_distribution
    )
end

function _get_single_beam_backscatter(filename, backscatter_data_directory)
    # Given a pitch angle and energy of an input electron beam, retrieves nearest precalculated backscatter distribution. Code 
    # adapted from https://github.com/GrantBerland/G4EPP/blob/main/G4EPP/examples/invert_ELFIN_measurements.ipynb and ported
    # to Julia by me.
    
    # Read datafile for this beam energy and pitch angle
    data = readdlm("$(backscatter_data_directory)/$(filename)", ',')
    
    # Extract momenta
    px = data[:,1]
    py = data[:,2]
    pz = data[:,3]

    # Calculate energy
    particle_energies = norm.(eachrow(data))

    # Calculate pitch angle
    # Remove energy from momentum direction (renormalize)
    px ./= particle_energies
    py ./= particle_energies
    pz ./= particle_energies

    # Get rotatation angle into magnetic reference frame at PFISR latitude
    untilt_angle = -(π + 12.682 * π / 180)
    # Rotate about x
    py = ( cos(untilt_angle) .* py - sin(untilt_angle) .* pz )
    pz = ( sin(untilt_angle) .* py + cos(untilt_angle) .* pz )
    # Calculate pitch angle
    particle_pitch_angles = rad2deg.(atan.(sqrt.(px.^2 + py.^2), pz))

    return _exact_2dhistogram(particle_energies, particle_pitch_angles, energy_bin_edges, pa_bin_edges) ./ 1e5 # To make units match with input
end

# ---------------- Plotting Functions ----------------
function plot_distribution(distribution)
    energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_data_bins()

    xlims = (0, 180)
    Δx = xlims[2] - xlims[1]

    ylims = log10.([10, 1e4])
    Δy = ylims[2] - ylims[1]

    heatmap(pa_bin_edges, log10.(energy_bin_edges), log10.(distribution),
        xlabel = "Pitch Angle, deg",
        xlims = xlims,

        ylabel = "Energy, keV",
        ylims = ylims,
        yticks = ([1, 2, 3, 4], ["10¹", "10²", "10³", "10⁴"]),

        colorbar_title = "\nLog10 # Electrons",
        colormap = :haline,

        aspect_ratio = Δx/Δy,
        size = (1.3, 1) .* 350,

        background_color_inside = :black,
        bordercolor = :transparent, # No axis border
        foreground_color_axis = :transparent, # No ticks
        framestyle = :box,
        grid = false,
        dpi = 300
    )
    display(plot!())
end

function individual_bounce_plots(distributions; noisegate = -Inf)
    energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_data_bins()

    n_plots = length(distributions[:,1,1])
    plots = []

    clims = (0, max(log10.(distributions[1,:,:])...))
    for i = 1:n_plots
        to_plot = distributions[i,:,:]
        to_plot[to_plot .<= noisegate] .= 0
        heatmap(pa_bin_edges, log10.(energy_bin_edges), log10.(to_plot),
            title = "$(i-1) Bounces",

            xlabel = "Pitch Angle, deg",
            xlims = (0, 180),

            ylabel = "Energy, keV",
            ylims = log10.((10, 10e3)),
            yticks = ([1, 2, 3, 4], ["10¹", "10²", "10³", "10⁴"]),

            colorbar_title = "Log10 # Electrons",
            clims = clims,
            colormap = :haline,

            leftmargin = 5mm,

            aspect_ratio = 180/log10.(10e3/10),
            background_color_inside = :black,
            framestyle = :box
        )
        e_pa_heatmap = plot!()

        # Get individual histograms
        energy_spectrum = dropdims(sum(distributions[i,:,:], dims = 2), dims = 2)
        append!(energy_spectrum, energy_spectrum[end]) # So that step plotting looks right
        plot(energy_bin_edges, energy_spectrum,
            title = "Energy",
            permute = (:y, :x),
            linetype = :steppost,
            label = false,

            xlims = (10, 10e3),
            xscale = :log10,

            ylims = (0, max(energy_spectrum...)),

            rightmargin = -6mm,
            aspect_ratio = max(energy_spectrum...)/(10e3-10)
        )
        energy = plot!()

        pitch_angle_spectrum = dropdims(sum(distributions[i,:,:], dims = 1), dims = 1)
        append!(pitch_angle_spectrum, pitch_angle_spectrum[end]) # So that step plotting looks right
        if iseven(i); reverse!(pitch_angle_spectrum); end # Northern hemisphere normalization
        plot(pa_bin_edges, pitch_angle_spectrum,
            title = "NH Pitch Angle",
            linetype = :steppost,
            label = false,

            xlims = (0, 180),

            ylims = (0, max(pitch_angle_spectrum...)),

            aspect_ratio = 180/max(pitch_angle_spectrum...)
        )
        pa = plot!()


        layout = @layout [a{.4w} b c]
        plot(e_pa_heatmap, energy, pa,
            layout = layout,
            size = (2,.75) .* 400,
            dpi = 300
        )

        push!(plots, plot!())
    end



    plot(plots...,
        layout = (n_plots, 1),
        size = (3, n_plots) .* 250,
        background = :transparent,
        dpi = 300
    )
    display("image/png",plot!())
end

function compare_input_to_output(distributions; remove_input = false, show_plot = true)
    input_distribution = distributions[1,:,:]

    # Calculate output distribution
    range_to_sum = 1:size(distributions)[1]
    if remove_input == true
        range_to_sum = 2:size(distributions)[1]
    end
    output_distribution = dropdims(sum(distributions[range_to_sum, :, :], dims = 1), dims = 1)

    e_ymax_in = max(sum(input_distribution, dims = 2)...) * 1.1
    e_ymax_out = max(sum(output_distribution, dims = 2)...) * 1.1
    e_ymax = e_ymax_in #max(e_ymax_in, e_ymax_out)

    pa_ymax_in = max(sum(input_distribution, dims = 1)...) * 1.1
    pa_ymax_out = max(sum(output_distribution, dims = 1)...) * 1.1
    pa_ymax = pa_ymax_in #max(pa_ymax_in, pa_ymax_out)

    clims = (log10.(max(input_distribution...)) - 4, log10.(max(input_distribution...)))

    in = _heatmap_and_spectra(input_distribution, e_ymax, pa_ymax, clims, "Simulation Input")
    out = _heatmap_and_spectra(output_distribution, e_ymax, pa_ymax, clims, "$(size(distributions)[1]-1) Bounces")

    plot(in, out,
        layout = (2,1),
        size = (2, 1.2) .* 500,
        background = :transparent,
        dpi = 250
    )
    if show_plot == true; display("image/png",plot!()); end
    return plot!()
end

function _heatmap_and_spectra(distribution, e_ymax, pa_ymax, clims, title)
    energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_data_bins()

    xlims = (0, 180)
    Δx = xlims[2] - xlims[1]

    ylims = log10.([10, 1e4])
    Δy = ylims[2] - ylims[1]

    heatmap(pa_bin_edges, log10.(energy_bin_edges), log10.(distribution),
        title = title,

        xlabel = "Pitch Angle, deg",
        xlims = xlims,

        ylabel = "Energy, keV",
        ylims = ylims,
        yticks = ([1, 2, 3, 4], ["10¹", "10²", "10³", "10⁴"]),

        colorbar_title = "Log10 # Electrons",
        clims = clims,
        colormap = :haline,

        aspect_ratio = Δx/Δy,


        topmargin = -5mm,
        bottommargin = -5mm,
        rightmargin = 5mm,
        leftmargin = 5mm,

        background_color_inside = :black,
        bordercolor = :transparent, # No axis border
        foreground_color_axis = :transparent, # No ticks
        framestyle = :box,
        grid = false
    )
    image = plot!()
    energy_spectrum = dropdims(sum(distribution, dims = 2), dims = 2)
    plot(energy_bin_means, energy_spectrum,
        title = "Energy Spectrum",

        linetype = :steppost,
        linecolor = RGB(0x00afda),
        linewidth = 1.4,
        label = false,

        xlabel = "Energy",
        xlims = (10, 10e3),
        xscale = :log10,

        ylabel = "# Electrons",
        ylims = (0, e_ymax), 
        
        aspect_ratio = (10e3-10)/e_ymax,

        topmargin = -5mm,
        bottommargin = -5mm,
        rightmargin = 5mm
    )
    energy = plot!()
    pa_spectrum = dropdims(sum(distribution, dims = 1), dims = 1)
    #ΔΩ = [2π * (cosd.(pa_bin_edges[i]) - cosd(pa_bin_edges[i+1])) for i = 1:pa_nbins]
    #pa_spectrum ./= ΔΩ # Normalize to particles per steradian
    # ^ not sure about that. it blows up the 0 and 180 regions
    
    plot(pa_bin_means, pa_spectrum,
        title = "Pitch Angle Spectrum",
        
        linetype = :steppost,
        linecolor = RGB(0x00afda),
        linewidth = 1.4,
        label = false,

        xlabel = "Pitch Angle",
        xlims = (0, 180),

        ylabel = "# Electrons",
        ylims = (0, pa_ymax), 

        aspect_ratio = 180/pa_ymax,

        topmargin = -5mm,
        bottommargin = -5mm,
        rightmargin = 5mm
    )
    pa = plot!()

    layout = @layout [a{.4w} b c]
    plot(image, energy, pa,
        layout = layout,
        size = (3.2, 1) .* 300,
        background = :transparent,
    )
    return plot!()
end

function plot_multibounce_statistics(distributions; show_plot = true)
    n_distros = size(distributions)[1]

    n_particles = sum.(eachslice(distributions, dims = 1))
    percent_remaining = (n_particles ./ n_particles[begin]) .* 100

    energy_distros = copy(distributions) .* 0
    energy_in_each_distro = zeros(n_distros)

    for i = 1:n_distros
        for e = 1:size(distributions)[2]
            energy_distros[i,e,:] = distributions[i,e,:] .* energy_bin_means[e]
        end
        energy_in_each_distro[i] = sum(energy_distros[i,:,:])
    end

    energy_deposited_cdf = cat(0, cumsum(-diff(energy_in_each_distro)), dims = 1)

    plot(0:n_distros-1, energy_deposited_cdf,
        title = "Energy Deposition",

        marker = true,
        markercolor = :black,

        linecolor = :black,
        linewidth = 1.2,
        label = false,

        xlabel = "Number of Bounces",

        ylabel = "Energy Deposited (keV)",
        
        margin = 5mm
    )
    energy = plot!()

    plot(0:n_distros-1, percent_remaining,
        title = "% Input Population Remaining",

        marker = true,
        markercolor = :black,

        linecolor = :black,
        linewidth = 1.2,
        label = false,

        xlabel = "Number of Bounces",

        ylabel = "% particles remaining",
        ylims = (.01, 120),
        yscale = :log10,

        margin = 5mm
    )
    particles = plot!()

    plot(energy, particles,
        layout = (1,2),
        background = :transparent,
        size = (2,1) .* 400,
        dpi = 300
    )
    display("image/png",plot!())
end

# ---------------- General Use Helpful Functions ----------------
function _print_progress_bar(fraction; bar_length = 20, overwrite = true)
# Prints a progress bar to terminal filled to user-specified percentage.
    character_length = 1/bar_length
    number_of_filled_characters = Int(floor(fraction/character_length))
    if overwrite == true; print("\r"); end
    print(repeat("█", number_of_filled_characters))
    print(repeat("░", bar_length - number_of_filled_characters))
    print(" [$(round(fraction*100, digits = 1))%]")
end

function _bin_edges_to_means(edges)
# Calculates means of bins given their edges.
    return [mean([edges[i], edges[i+1]]) for i = 1:length(edges)-1] 
end

function _exact_2dhistogram(x, y, x_bin_edges, y_bin_edges; weights = ones(length(x)))
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
