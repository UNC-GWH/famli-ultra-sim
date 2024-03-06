% Simulating B-mode Ultrasound Images Example
%
% This example illustrates how k-Wave can be used for the simulation of
% B-mode ultrasound images using a phased-array or sector transducer. It
% builds on the Simulating B-mode Ultrasound Images Example.
%
% To allow the simulated scan line data to be processed multiple times with
% different settings, the simulated RF data is saved to disk. This can be
% reloaded by setting RUN_SIMULATION = false within the example m-file. The
% data can also be downloaded from
% http://www.k-wave.org/datasets/example_us_phased_array_scan_lines.mat 
%
% author: Bradley Treeby
% date: 7th September 2012
% last update: 22nd January 2020
%  
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2012-2020 Bradley Treeby

% This file is part of k-Wave. k-Wave is free software: you can
% redistribute it and/or modify it under the terms of the GNU Lesser
% General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% 
% k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
% more details. 
% 
% You should have received a copy of the GNU Lesser General Public License
% along with k-Wave. If not, see <http://www.gnu.org/licenses/>. 

%#ok<*UNRCH>

function us_bmode_phased_array(sound_speed_map_fn, density_map_fn, output_fn)
    % simulation settings
    DATA_CAST       = 'single';     % set to 'single' or 'gpuArray-single' to speed up computations

    tmp_dir = fullfile(tempdir, matlab.lang.internal.uuid());
    if not(isfolder(tmp_dir))
        mkdir(tmp_dir);
    end
    clear tempdir;
    setenv('TMPDIR', tmp_dir);

    % =========================================================================
    % DEFINE THE K-WAVE GRID
    % =========================================================================

    sound_speed_map = nrrdread(sound_speed_map_fn);
    density_map = nrrdread(density_map_fn);

    sound_speed_map_info = nrrdinfo(sound_speed_map_fn);
    
    % set the size of the perfectly matched layer (PML)
    pml_x_size = 10;                % [grid points]
    pml_y_size = 10;                % [grid points]
    pml_z_size = 10;                % [grid points]
    
    % set total number of grid points not including the PML
    sc = 1;
    Nx = sound_speed_map_info.ImageSize(1);     % [grid points]
    Ny = sound_speed_map_info.ImageSize(2);     % [grid points]
    Nz = sound_speed_map_info.ImageSize(3);     % [grid points]

    % 0.0009892949019558756, 0.0009892949019559938, 0.000989294901955991
    dx = sound_speed_map_info.PixelDimensions(1)/4.0;                    % [m]
    dy = sound_speed_map_info.PixelDimensions(2)/4.0;                    % [m]
    dz = sound_speed_map_info.PixelDimensions(3)/4.0;                    % [m]

    disp(["size", Nx, Ny, Nz])
    disp(["spc", dx, dy, dz])

    % dx = 0.00045
    % dy = 0.00045
    % dz = 0.00045

    % x = 50e-3;                      % [m]
    % % calculate the spacing between the grid points
    % dx = x / Nx;                    % [m]
    % dy = dx;                        % [m]
    % dz = dx;   
    
    % create the k-space grid
    kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);
    
    % =========================================================================
    % DEFINE THE MEDIUM PARAMETERS
    % =========================================================================
    
    % define the properties of the propagation medium
    c0 = 1540;                      % [m/s]
    rho0 = 1000;                    % [kg/m^3]
    medium.alpha_coeff = 0.75;      % [dB/(MHz^y cm)]
    medium.alpha_power = 1.1;
    medium.BonA = 6;
    
    % create the time array
    t_end = (Nx * dx) * 10.0 / c0;   % [s]
    disp(["t_end", t_end])
    kgrid.makeTime(c0, [], t_end);
    
    % =========================================================================
    % DEFINE THE INPUT SIGNAL
    % =========================================================================
    
    % define properties of the input signal
    % source_strength = 0.718e3;          % [Pa]
    % tone_burst_freq = 2.19e3;     % [Hz]
    source_strength = 1e6;          % [Pa]
    tone_burst_freq = 1e6;     % [Hz]
    tone_burst_cycles = 4;
    
    % create the input signal using toneBurst 
    input_signal = toneBurst(1/kgrid.dt, tone_burst_freq, tone_burst_cycles);
    % disp(input_signal);
    
    % % scale the source magnitude by the source_strength divided by the
    % % impedance (the source is assigned to the particle velocity)
    % input_signal = (source_strength ./ (c0 * rho0)) .* input_signal;
    % disp(input_signal);
    
    % =========================================================================
    % DEFINE THE ULTRASOUND TRANSDUCER
    % =========================================================================
    
    % define the physical properties of the phased array transducer
    transducer.number_elements = 64;       % total number of transducer elements
    transducer.element_width = 1;               % width of each element [grid points]
    transducer.element_length = 40;        % length of each element [grid points]
    transducer.element_spacing = 0;             % spacing (kerf  width) between the elements [grid points]
    
    % calculate the width of the transducer in grid points
    transducer_width = transducer.number_elements * transducer.element_width ...
        + (transducer.number_elements - 1) * transducer.element_spacing;
    
    % use this to position the transducer in the middle of the computational grid
    transducer.position = round([1, Ny/2 - transducer_width/2, Nz/2 - transducer.element_length/2]);
    
    % properties used to derive the beamforming delays
    transducer.sound_speed = c0;                    % sound speed [m/s]
    transducer.focus_distance = 30e-3;              % focus distance [m]
    transducer.elevation_focus_distance = 30e-3;    % focus distance in the elevation plane [m]
    transducer.steering_angle = 0;                  % steering angle [degrees]
    transducer.steering_angle_max = 32;             % maximum steering angle [degrees]
    
    % apodization
    transducer.transmit_apodization = 'Hanning';    
    transducer.receive_apodization = 'Rectangular';
    
    % define the transducer elements that are currently active
    transducer.active_elements = ones(transducer.number_elements, 1);
    
    % append input signal used to drive the transducer
    transducer.input_signal = input_signal;
    
    % create the transducer using the defined settings
    transducer = kWaveTransducer(kgrid, transducer);
    
    % print out transducer properties
    transducer.properties;
    
    % =========================================================================
    % DEFINE THE MEDIUM PROPERTIES
    % =========================================================================
    
    % % define a random distribution of scatterers for the medium
    % background_map_mean = 1;
    % background_map_std = 0.008;
    % background_map = background_map_mean + background_map_std * randn([Nx, Ny, Nz]);
    % 
    % % define a random distribution of scatterers for the highly scattering
    % % region
    % scattering_map = randn([Nx, Ny, Nz]);
    % scattering_c0 = c0 + 25 + 75 * scattering_map;
    % scattering_c0(scattering_c0 > 1600) = 1600;
    % scattering_c0(scattering_c0 < 1400) = 1400;
    % scattering_rho0 = scattering_c0 / 1.5;
    % 
    % % define properties
    % sound_speed_map = c0 * ones(Nx, Ny, Nz) .* background_map;
    % density_map = rho0 * ones(Nx, Ny, Nz) .* background_map;
    % 
    % % define a sphere for a highly scattering region
    % radius = 8e-3;
    % x_pos = 32e-3;
    % y_pos = dy * Ny/2;
    % scattering_region1 = makeBall(Nx, Ny, Nz, round(x_pos / dx), round(y_pos / dx), Nz/2, round(radius / dx));
    % 
    % % assign region
    % sound_speed_map(scattering_region1 == 1) = scattering_c0(scattering_region1 == 1);
    % density_map(scattering_region1 == 1) = scattering_rho0(scattering_region1 == 1);
    
    sound_speed_map = sound_speed_map - min(sound_speed_map(:));
    sound_speed_map = sound_speed_map / max(sound_speed_map(:));
    sound_speed_map = sound_speed_map * (1601.5913243064965 - 1400.0);
    sound_speed_map = sound_speed_map + 1400.0;
    
    density_map = density_map - min(density_map(:));
    density_map = density_map / max(density_map(:));
    density_map = density_map * (1066.6666666666667 - 933.3333333333334);
    density_map = density_map + 933.3333333333334;
    
    % assign to the medium inputs
    medium.sound_speed = sound_speed_map;
    medium.density = density_map;
    
    % =========================================================================
    % RUN THE SIMULATION
    % =========================================================================
    
    % range of steering angles to test
    steering_angles = -32:2:32;
    
    % preallocate the storage
    number_scan_lines = length(steering_angles);
    scan_lines = zeros(number_scan_lines, kgrid.Nt);
    
    % set the input settings
    input_args = {...
        'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size, pml_z_size], ...
        'DataCast', DATA_CAST, 'DataRecast', true, 'PlotSim', false};
        
    % loop through the range of angles to test
    for angle_index = 1:number_scan_lines
        
        % update the command line status
        disp('');
        disp(['Computing scan line ' num2str(angle_index) ' of ' num2str(number_scan_lines)]);

        % update the current steering angle
        transducer.steering_angle = steering_angles(angle_index);
        
        % run the simulation
        sensor_data = kspaceFirstOrder3DG(kgrid, medium, transducer, transducer, input_args{:});

        % if isnan(sensor_data)
        %     sensor_data(isnan(sensor_data)) = 0;
        % end
        % extract the scan line from the sensor data
        scan_lines(angle_index, :) = transducer.scan_line(sensor_data);
        
    end

    disp(size(scan_lines));
    % trim the delay offset from the scan line data
    t0_offset = round(length(input_signal) / 2) + (transducer.appended_zeros - transducer.beamforming_delays_offset);
    scan_lines = scan_lines(:, t0_offset:end);
    disp(size(scan_lines));
    
    % get the new length of the scan lines
    Nt = length(scan_lines(1, :));
    
    % =========================================================================
    % PROCESS THE RESULTS
    % =========================================================================
    
    % -----------------------------
    % Remove Input Signal
    % -----------------------------
    
    % create a window to set the first part of each scan line to zero to remove
    % interference from the input signal
    scan_line_win = getWin(Nt * 2, 'Tukey', 'Param', 0.05).';
    scan_line_win = [zeros(1, t0_offset * 2), scan_line_win(1:end/2 - t0_offset * 2)];
    
    % apply the window to each of the scan lines
    disp(size(scan_line_win));
    disp(size(scan_lines));
    scan_lines = bsxfun(@times, scan_line_win, scan_lines);
    
    % -----------------------------
    % Time Gain Compensation
    % -----------------------------
    
    % create radius variable
    r = c0 * (1:Nt) * kgrid.dt / 2;    % [m]
    
    % define absorption value and convert to correct units
    tgc_alpha_db_cm = medium.alpha_coeff * (tone_burst_freq * 1e-6)^medium.alpha_power;
    tgc_alpha_np_m = tgc_alpha_db_cm / 8.686 * 100;
    
    % create time gain compensation function based on attenuation value and
    % round trip distance
    tgc = exp(tgc_alpha_np_m * 2 * r);
    
    % apply the time gain compensation to each of the scan lines
    scan_lines = bsxfun(@times, tgc, scan_lines);
    
    % -----------------------------
    % Frequency Filtering
    % -----------------------------
    
    % filter the scan lines using both the transmit frequency and the second
    % harmonic
    scan_lines_fund = gaussianFilter(scan_lines, 1/kgrid.dt, tone_burst_freq, 100, true);
    scan_lines_harm = gaussianFilter(scan_lines, 1/kgrid.dt, 2 * tone_burst_freq, 30, true);
    
    % -----------------------------
    % Envelope Detection
    % -----------------------------
    
    % envelope detection
    scan_lines_fund = envelopeDetection(scan_lines_fund);
    scan_lines_harm = envelopeDetection(scan_lines_harm);
    
    % -----------------------------
    % Log Compression
    % -----------------------------
    
    % normalised log compression
    compression_ratio = 3;
    scan_lines_fund = logCompression(scan_lines_fund, compression_ratio, true);
    scan_lines_harm = logCompression(scan_lines_harm, compression_ratio, true);
    
    % -----------------------------
    % Scan Conversion
    % -----------------------------
    
    % set the desired size of the image
    image_size = [Nx * dx, Ny * dy];
    
    % convert the data from polar coordinates to Cartesian coordinates for
    % display
    b_mode_fund = scanConversion(scan_lines_fund, steering_angles, image_size, c0, kgrid.dt);
    b_mode_harm = scanConversion(scan_lines_harm, steering_angles, image_size, c0, kgrid.dt);



    save(output_fn, "scan_lines", "scan_lines_fund", "b_mode_fund", "b_mode_harm");
    rmdir(tmp_dir);

    return;
    
end