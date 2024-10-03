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
include("./General_Functions.jl") # Provides general-purpose functions I find useful

# ---------------- Backscatter Simulation Functions ----------------
function multibounce_simulation(input_distribution, n_bounces; show_progress = false)
    energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_simulation_bins()
    @assert size(input_distribution) == (energy_nbins, pa_nbins) "Multibounce simulation requires input to be on simulation grid (E = $energy_nbins bins, α = $pa_nbins bins). Use set_simulation_bins() to change these bins."

    distributions = zeros(n_bounces + 1, energy_nbins, pa_nbins)
    distributions[1,:,:] = input_distribution
    for i = 2:(n_bounces+1)
        if show_progress == true; println("Bounce $(i-1)/$(n_bounces)"); end
        input_distribution = distributions[i-1,:,:]

        if i > 2 # Only do this for distributions after the first
            reverse!(input_distribution, dims = 2) # Move the distribution from the antiloss cone to the loss cone
        end

        distributions[i,:,:], loc, str = simulate_NH_backscatter(energy_bin_edges, pa_bin_edges, input_distribution, show_progress = show_progress, show_error = true, return_beam_strengths = true)
        show_beams(loc,str)
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

function simulate_NH_backscatter(e_bin_edges, pa_bin_edges, input_flux; return_beam_strengths = false, show_error = false, show_progress = false)
    # e_bin_edges : description [keV]
    # pa_bin_edges : description [deg]
    # input_flux : ..... [1/(MeV str) * <arb>] for ELFIN, <arb> is #/(cm2)

    # Guard block
    @assert size(input_flux) == (length(e_bin_edges)-1, length(pa_bin_edges)-1) "Bin numbers $((length(e_bin_edges)-1, length(pa_bin_edges)-1)) do not match flux distribution size $(size(input_flux))"

    # Calculate bin means
    e_bin_means = edges_to_means(e_bin_edges)
    pa_bin_means = edges_to_means(pa_bin_edges)

    # Calculate number of particles (times <arb>) in each pixel of input distribution
    ΔE = [(e_bin_edges[e+1] - e_bin_edges[e]) ./ 1000 for e in 1:length(e_bin_means)] # !!! ΔE IN MeV, E_BIN_EDGES IN keV !!!
    ΔΩ = [2π * (cosd(pa_bin_edges[α]) - cosd(pa_bin_edges[α+1])) for α in 1:length(pa_bin_means)]
    n_particles_input = [input_flux[e,α] * ΔE[e] * ΔΩ[α] for e in 1:length(e_bin_means), α in 1:length(pa_bin_means)]

    # Load precalculated beams
    backscatter_directory = "$(BackscatterSimulation_TOP_LEVEL)/data/binned_backscatter_distributions"
    backscatter_filepaths = glob("*deg.npz", backscatter_directory)
    backscatter_filenames = replace.(backscatter_filepaths, "$(backscatter_directory)/" => "")

    backscatter_energies = match.(r"(.*?)keV_", backscatter_filenames)
    backscatter_energies = [backscatter_energies[i].captures[1] for i = eachindex(backscatter_energies)]
    backscatter_energies = parse.(Int64, backscatter_energies)

    backscatter_pitch_angles = match.(r"keV_(.*?)deg.npz", backscatter_filenames)
    backscatter_pitch_angles = [backscatter_pitch_angles[i].captures[1] for i = eachindex(backscatter_pitch_angles)]
    backscatter_pitch_angles = parse.(Int64, backscatter_pitch_angles)

    # Preallocate result arrays
    result_energy_nbins, _, _, result_pa_nbins, _, _, SIMULATION_α_MAX = get_simulation_bins()
    output_distribution = zeros(result_energy_nbins, result_pa_nbins)
    n_particles_fit = copy(n_particles_input) .* 0
    beam_coordinates = _get_beam_locations() # (E, pa)
    beam_strengths = zeros(length(beam_coordinates))

    # Simulate backscatter
    pixels_to_simulate = findall(input_flux .≠ 0)
    for i = 1:length(pixels_to_simulate)
        # Print progress to terminal every so often
        if show_progress && ((i % 50 == 0) || (i == length(pixels_to_simulate)))
            print_progress_bar(i/length(pixels_to_simulate))
        end

        # Get energy and pitch angle of this pixel
        pixel_coords = pixels_to_simulate[i]
        e = e_bin_means[pixel_coords[1]] # Pixel energy
        pa = pa_bin_means[pixel_coords[2]] # Pixel pitch angle

        # Find nearest precalculated beam for this pixel
        energy_differences = abs.(backscatter_energies .- e)
        mindistance, _ = findmin(energy_differences)
        energy_idxs = findall(energy_differences .== mindistance)

        pa_differences = abs.(backscatter_pitch_angles .- pa)
        mindistance, _ = findmin(pa_differences)
        pa_idxs = findall(pa_differences .== mindistance)

        nearest_beam_idx = intersect(energy_idxs, pa_idxs)
        nearest_beam_idx = nearest_beam_idx[1] # Extract index from 1-element vector. If we are exactly halfway between two or more beams, this will round down

        nearest_e = backscatter_energies[nearest_beam_idx]
        nearest_pa = backscatter_pitch_angles[nearest_beam_idx]

        # Get backscatter
        beam_backscatter = npzread("$(backscatter_directory)/$(backscatter_filenames[nearest_beam_idx])")
        beam_backscatter = beam_backscatter["backscatter_distribution"]

        # Zero out regions where we don't have precalculated beams
        if ((0 <= pa <= SIMULATION_α_MAX) == false) || ((5 <= e <= 15e3) == false)
            beam_backscatter .*= 0
        end

        # If the process of snapping the pixel to the nearest beam moves its energy bin,
        # adjust n_particles accordingly to make energy integral equal for input and fit
        old_e_idx = findlast(e .≥ e_bin_edges[1:end-1])
        new_e_idx = findlast(nearest_e .≥ e_bin_edges[1:end-1])
        n_particles_fit[pixel_coords] = n_particles_input[pixel_coords] * (e_bin_means[old_e_idx] / e_bin_means[new_e_idx])

        # Add pixel to beam strength arrays
        beam_idx = findfirst(beam_coordinates .== [(nearest_e, nearest_pa)])
        beam_strengths[beam_idx] += n_particles_fit[pixel_coords]

        # Add backscatter to result
        output_distribution .+= beam_backscatter .* n_particles_fit[pixel_coords]
    end
    if show_progress == true; println(); end

    # Calculate fitting error - i.e amount of energy gained or lost by process of snapping each input pixel to a beam that may be at a different energy/pitch angle
    if show_error == true
        fitted_flux = beams_to_flux(e_bin_edges, pa_bin_edges, beam_coordinates, beam_strengths)
        fitted_total_energy = sum([beam_strengths[i] * beam_coordinates[i][1] for i in eachindex(beam_strengths)])
        fitted_total_particles = sum(n_particles_fit)

        input_culled = input_flux[:, pa_bin_means .< SIMULATION_α_MAX]
        input_total_energy = sum([n_particles_input[e,α]*e_bin_means[e] for e in 1:length(e_bin_means), α in 1:length(pa_bin_means)])
        input_total_particles = sum(n_particles_input)
        
        energy_fitting_error = fitted_total_energy / input_total_energy
        println("fit energy = $(round(energy_fitting_error, sigdigits = 3)) x input energy")

        particle_fitting_error = fitted_total_particles / input_total_particles
        println("fit particles = $(round(particle_fitting_error, sigdigits = 3)) x input particles")
    end
    
    # Return
    if return_beam_strengths == true
        return output_distribution, beam_coordinates, beam_strengths
    else
        return output_distribution
    end
end

function atmosphere_loss_rate(distributions)
    # Use least-squares regression to estimate loss rate of particles to atomsphere in a multibounce distribution
    n_particles = sum.(eachslice(distributions, dims = 1))

    if length(n_particles) == 1; return NaN; end

    A = hcat(1:length(n_particles), ones(length(n_particles))) # Matrix A for least squares problem y = Ac
    remaining_factor_logspace, _ = A \ log10.(n_particles)

    loss_rate = 1 - (10^remaining_factor_logspace)
    return loss_rate
end

function _get_beam_locations()
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

    # Return
    return collect(zip(backscatter_energies, backscatter_pitch_angles))
end

function get_simulation_bins()
    data_bins = npzread("$(BackscatterSimulation_TOP_LEVEL)/data/binned_backscatter_distributions/data_bins.npz")

    energy_nbins = data_bins["energy_nbins"]
    energy_bin_edges = data_bins["energy_bin_edges"]
    energy_bin_means = data_bins["energy_bin_means"]

    pa_nbins = data_bins["pa_nbins"]
    pa_bin_edges = data_bins["pa_bin_edges"]
    pa_bin_means = data_bins["pa_bin_means"]

    SIMULATION_α_MAX = 77.5

    return energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX
end

# ---------------- Backscatter Binning Functions ----------------
function set_simulation_bins(; energy_nbins = 35, pa_nbins = 100, debug = false)
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
        print_progress_bar(i/length(backscatter_filenames))
        _prebake_backscatter_file(backscatter_filenames[i], backscatter_data_directory)
    end
    println("\n")

    # Show debug info if needed
    if debug == true
        # Show all input beams
        energies = [parse(Int64, match.(r"bs_spectra(.*?)keV", backscatter_filenames[i])[1]) for i = eachindex(backscatter_filenames)]
        pitch_angles = [parse(Int64, match.(r"PAD(.*?).csv", backscatter_filenames[i])[1]) for i = eachindex(backscatter_filenames)]

        plot(
            title = "Input Beams",

            xlabel = "Pitch Angle, deg",
            xlims = (0, 180),

            ylabel = "Energy, keV",
            ylims = (10, 1e4),
            yscale = :log10,

            grid = false
        )
        hline!(energy_bin_edges, linecolor = RGB(.8,.8,.8), label = false)
        vline!(pa_bin_edges, linecolor = RGB(.8,.8,.8), label = false)
        scatter!(pitch_angles, energies, label = false, color = :black)
        display(plot!())
    end
end

function _prebake_backscatter_file(filename, backscatter_data_directory)
    energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_simulation_bins()

    # Get beam parameters from filename
    energy = match.(r"bs_spectra(.*?)keV", filename)[1]
    energy = parse(Int64, energy)

    pitch_angle = match.(r"PAD(.*?).csv", filename)[1]
    pitch_angle = parse(Int64, pitch_angle)

    backscatter_distribution = _get_single_beam_backscatter(filename, backscatter_data_directory, energy_bin_edges, pa_bin_edges)

    npzwrite("$(BackscatterSimulation_TOP_LEVEL)/data/binned_backscatter_distributions/$(energy)keV_$(pitch_angle)deg.npz",
        energy_bin_edges = energy_bin_edges,
        pitch_angle_bin_edges = pa_bin_edges,
        backscatter_distribution = backscatter_distribution
    )
end

function _get_single_beam_backscatter(filename, backscatter_data_directory, energy_bin_edges, pa_bin_edges)
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

    return _exact_2dhistogram(particle_energies, particle_pitch_angles, energy_bin_edges, pa_bin_edges) ./ 1e5 # /1e5 to make units match with input
end

# ---------------- Plotting Functions ----------------
function plot_distribution(distribution)
    energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_simulation_bins()

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

function show_beams(locations, strengths; clims = (0, max(log10.(strengths)...)))
    data_color = replace(log10.(strengths), -Inf => -100) # -Inf breaks the zcolor argument
    scatter(beams,
        label = false,
        zcolor = data_color,
        colormap = :magma,
        markerstrokewidth = 0,

        title = "Input Beams",

        xlabel = "Energy, keV",
        xscale = :log10,

        ylabel = "Pitch Angle, deg",
        ylims = (0, 180),

        colorbar_title = "Log10 Beam Strength",
        clims = clims,

        permute = (:y, :x), # Swap x & y axes
        background_color_inside = :black,
        grid = false,
        dpi = 300
    )
    display(plot!())
end

function individual_bounce_plots(distributions; noisegate = -Inf)
    energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_simulation_bins()

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
    display(plot!())
end

function compare_input_to_output(distributions; show_plot = true)
    simulation_energy_nbins, simulation_energy_bin_edges, simulation_energy_bin_means, simulation_pa_nbins, simulation_pa_bin_edges, simulation_pa_bin_means, SIMULATION_α_MAX = get_simulation_bins()

    n_bounces = size(distributions)[1] - 1
    
    input = distributions[1,:,:]
    input_energy_spectrum = dropdims(sum(input, dims = 2), dims = 2)
    input_pa_spectrum = dropdims(sum(input, dims = 1), dims = 1)

    output = dropdims(sum(distributions, dims = 1), dims = 1)
    output_energy_spectrum = dropdims(sum(output, dims = 2), dims = 2)
    output_pa_spectrum = dropdims(sum(output, dims = 1), dims = 1)

    # Input heatmap
    xlims = (0, 180)
    Δx = xlims[2] - xlims[1]

    ylims = log10.([10, 1e4])
    Δy = ylims[2] - ylims[1]

    cmax = log10(max(input[input .≠ 0]..., output[output .≠ 0]...))
    clims = (cmax-4, cmax)
    heatmap(simulation_pa_bin_edges, log10.(simulation_energy_bin_edges), log10.(input),
        title = "Input",

        xlabel = "Pitch Angle, deg",
        xlims = xlims,

        ylabel = "Energy, keV",
        ylims = ylims,
        yticks = ([1, 2, 3, 4], ["10¹", "10²", "10³", "10⁴"]),

        colorbar_title = "Log10 # Electrons",
        clims = clims,
        colormap = :haline,

        leftmargin = 10mm,

        aspect_ratio = Δx/Δy,
        background_color_inside = :black,
        bordercolor = :transparent, # No axis border
        foreground_color_axis = :transparent, # No ticks
        framestyle = :box,
        grid = false
    )
    input_heatmap = plot!()

    # Output heatmap
    heatmap(simulation_pa_bin_edges, log10.(simulation_energy_bin_edges), log10.(output),
        title = "$(n_bounces) Bounce Output",

        xlabel = "Pitch Angle, deg",
        xlims = xlims,

        ylabel = "Energy, keV",
        ylims = ylims,
        yticks = ([1, 2, 3, 4], ["10¹", "10²", "10³", "10⁴"]),

        colorbar_title = "Log10 # Electrons",
        clims = clims,
        colormap = :haline,

        aspect_ratio = Δx/Δy,
        background_color_inside = :black,
        bordercolor = :transparent, # No axis border
        foreground_color_axis = :transparent, # No ticks
        framestyle = :box,
        grid = false
    )
    output_heatmap = plot!()

    # Energy spectrum
    ymax = max(input_energy_spectrum...) * 1.2 #max(cat(input_energy_spectrum, output_energy_spectrum, dims = 1)...)*1.1
    plot(simulation_energy_bin_means, input_energy_spectrum,
        title = "Energy Spectrum",

        linetype = :steppost,
        linecolor = :gray,
        linestyle = :dot,
        linewidth = 1.4,
        label = "Input",

        xlabel = "Energy",
        xlims = (10, 10e3),
        xscale = :log10,

        ylabel = "# Electrons",
        ylims = (0, ymax),

        topmargin = 5mm,
        
        aspect_ratio = (10e3-10)/ymax,
        framestyle = :box,
        tickdirection = :out,
        legend = true,
    )
    plot!(simulation_energy_bin_means, output_energy_spectrum,
        linetype = :steppost,
        linecolor = :black,
        linestyle = :solid,
        linewidth = 1.4,
        label = "Output",

    )
    e_plot = plot!()

    # PA spectrum
    ymax = max(input_pa_spectrum...) * 1.2 #max(cat(input_pa_spectrum, output_pa_spectrum, dims = 1)...)*1.1
    plot(simulation_pa_bin_means, input_pa_spectrum,
        title = "Pitch Angle Spectrum",
                
        linetype = :steppost,
        linecolor = :gray,
        linestyle = :dot,
        linewidth = 1.4,
        label = "Input",

        xlabel = "Pitch Angle",
        xlims = (0, 180),

        ylabel = "# Electrons",
        ylims = (0, ymax),

        topmargin = 5mm,

        aspect_ratio = 180/ymax,
        framestyle = :box,
        tickdirection = :out,
        legend = false,
    )
    plot!(simulation_pa_bin_means, output_pa_spectrum,
        linetype = :steppost,
        linecolor = :black,
        linestyle = :solid,
        linewidth = 1.4,
        label = "Output",
    )
    pa_plot = plot!()

    # Multibounce statistics
    multibounce = _multibounce_statistics_plot(distributions)

    # Main plot
    layout = @layout [a b
                      c d
                      e]
    plot(input_heatmap, output_heatmap, e_plot, pa_plot, multibounce,
        layout = layout,
        background = :transparent,
        size = (1, 1.25) .* 700,
        dpi = 300
    )
    if show_plot == true; display(plot!()); end
    return plot!()
end

function plot_percent_change(distributions)
    energy_nbins, energy_bin_edges, energy_bin_means, pa_nbins, pa_bin_edges, pa_bin_means, SIMULATION_α_MAX = get_simulation_bins()
    loss_cone_slice = pa_bin_edges[begin:end-1] .< SIMULATION_α_MAX

    input = distributions[1,:,:]
    output = dropdims(sum(distributions, dims = 1), dims = 1)

    fraction_change = (output .- input) ./ input
    percent_change = fraction_change .* 100

    if any(percent_change .< 0); error("something is deeply wrong. pray for forgiveness."); end

    heatmap(pa_bin_edges, log10.(energy_bin_edges), percent_change,
        xlabel = "Pitch Angle, deg",
        xlims = (0, SIMULATION_α_MAX),

        ylabel = "Energy, eV",
        ylims = (2, 4),
        yticks = ([2, 3, 4], ["10²", "10³", "10⁴"]),

        aspect_ratio = (180/2),

        colorbar_title = "Percent Increase",
        colormap = cgrad(:cherry, rev = true),
        clims = (0, 100),

        grid = false,
        framestyle = :box,
        background_color_outside = :transparent,
        background_color_inside = :white,
        dpi = 300
    )
    display(plot!())
end

function _multibounce_statistics_plot(distributions)
    simulation_energy_nbins, simulation_energy_bin_edges, simulation_energy_bin_means, simulation_pa_nbins, simulation_pa_bin_edges, simulation_pa_bin_means, SIMULATION_α_MAX = get_simulation_bins()

    n_distros = size(distributions)[1]

    n_particles = sum.(eachslice(distributions, dims = 1))
    percent_remaining = (n_particles ./ n_particles[begin]) .* 100

    energy_distros = copy(distributions) .* 0
    energy_in_each_distro = zeros(n_distros)

    for i = 1:n_distros
        for e = 1:size(distributions)[2]
            energy_distros[i,e,:] = distributions[i,e,:] .* simulation_energy_bin_means[e]
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
        xlims = (0, n_distros-.9),

        ylabel = "Energy Deposited (keV)",
        ylims = (0, energy_deposited_cdf[end]*1.1),

        tickdirection = :out,
        framestyle = :box,
        aspect_ratio = (n_distros-.8)/(energy_deposited_cdf[end]*1.1)
    )
    energy = plot!()

    plot(0:n_distros-1, percent_remaining,
        title = "% Input Remaining",

        marker = true,
        markercolor = :black,

        linecolor = :black,
        linewidth = 1.2,
        label = false,

        xlabel = "Number of Bounces",
        xlims = (-.1, n_distros-.9),

        ylabel = "% particles remaining",
        ylims = (.01, 130),
        yscale = :log10,

        tickdirection = :out,
        framestyle = :box,
        aspect_ratio = (n_distros-.8)/(130-.1)
    )
    particles = plot!()

    plot(energy, particles,
        layout = (1,2),
        background = :transparent
    )
    return plot!()
end