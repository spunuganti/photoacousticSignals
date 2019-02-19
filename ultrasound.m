clear all

%%Details of ulatrasound data collection
% 1. tissue density used= 1040 and no of samples=4001
% 2. tissue density used=1059 and no of samples=1001 and transducer
% tone_burst_freq to 0.25e6 from 0.5e6 and transducer.focus_distance =
% 22e-3 from 20e-3
% 3. transducer.focus_distance = 18e-3;  from 22e-3            % focus distance [m]
%transducer.elevation_focus_distance = 17e-3; from 19e-3   % focus distance in the elevation plane [m]
%medium.density=950 for adipose from 1049?
% receiver apodization = hanning from rectangular
% tone_burst_freq = 1e6; from 0.25e6        % [Hz]
% tone_burst_cycles = 7; from 5

%4 transducer.transmit_apodization = 'Hanning';  from rectangular
% source_strength = 2e6;          % [MPa]
% tone_burst_freq = 0.5e6;        % [Hz]
% medium.density=975; 
% transducer.focus_distance = 20e-3;              % focus distance [m]
% transducer.elevation_focus_distance = 17e-3;    % focus distance in the elevation plane [m]
% transducer.active_elements(10:60) = 1;
% transducer.steering_angle = 2;     from 0             % steering angle [degrees]


% size of the computational grid
Nx = 120;    % number of grid points in the x (row) direction
x = 60e-3;   % size of the domain in the x direction [m]
dx = x/Nx;  % grid point spacing in the x direction [m]

Ny = 120;    % number of grid points in the x (row) direction
y = 60e-3;   % size of the domain in the x direction [m]
dy = y/Ny;  % grid point spacing in the x direction [m]

Nz = 120;    % number of grid points in the x (row) direction
z = 60e-3;   % size of the domain in the x direction [m]
dz = z/Nz;  % grid point spacing in the x direction [m]

% time array
dt = 25e-9;                      % [s]
t_end = 2.5e-5;                 % [s] 1000 samples

% create the computational grid
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);

% create the time array
kgrid.setTime(round(t_end / dt) + 1, dt);


% define a single sensor point
sensor.mask = zeros(Nx, Ny, Nz);
sensor.mask(:, :, Nz/2) = 1;
%sensor.mask(Nx/2 - source_sensor_distance, Ny/2, Nz/2) = 1;


% computation settings
input_args = {'DataCast', 'single'};

% define properties of the input signal
source_strength = 4e6;          % [MPa]
tone_burst_freq = 2e6;        % [Hz]
tone_burst_cycles = 10;

% define the properties of the propagation medium
medium.sound_speed = 1560;      % [m/s]
medium.density = 1040; 

% create the input signal using toneBurst
input_signal = toneBurst(1/kgrid.dt, tone_burst_freq, tone_burst_cycles);

% scale the source magnitude by the source_strength divided by the
% impedance (the source is assigned to the particle velocity)
input_signal = (source_strength ./ (medium.sound_speed * medium.density)) .* input_signal;

% physical properties of the transducer
transducer.number_elements = 72;    % total number of transducer elements
transducer.element_width = 1;       % width of each element [grid points]
transducer.element_length = 12;     % length of each element [grid points]
transducer.element_spacing = 0;     % spacing (kerf width) between the elements [grid points]
transducer.radius = inf;            % radius of curvature of the transducer [m]

% calculate the width of the transducer in grid points
transducer_width = transducer.number_elements * transducer.element_width ...
    + (transducer.number_elements - 1) * transducer.element_spacing;

% use this to position the transducer in the middle of the computational grid
transducer.position = round([1, Ny/2 - transducer_width/2, Nz/2 - transducer.element_length/2]);

% properties used to derive the beamforming delays
transducer.sound_speed = 1540;                  % sound speed [m/s]
transducer.focus_distance = 20e-3;              % focus distance [m]
transducer.elevation_focus_distance = 18e-3;    % focus distance in the elevation plane [m]
transducer.steering_angle = 1;                  % steering angle [degrees]

% apodization
transducer.transmit_apodization = 'Gaussian';    
transducer.receive_apodization = 'Gaussian';

% define the transducer elements that are currently active
transducer.active_elements = zeros(transducer.number_elements, 1);
transducer.active_elements(21:52) = 1;

% append input signal used to drive the transducer
transducer.input_signal = input_signal;

% create the transducer using the defined settings
transducer = kWaveTransducer(kgrid, transducer);

% run the simulation
[sensor_data] = kspaceFirstOrder3D(kgrid, medium, transducer, sensor, input_args{:});

% extract a single scan line from the sensor data using the current
% beamforming settings
scan_line = transducer.scan_line(sensor_data);

%% 
% define a sensor mask through the central plane of the transducer
sensor.mask = zeros(Nx, Ny, Nz);
sensor.mask(:, :, Nz/2) = 1;

% set the record mode such that only the rms and peak values are stored
sensor.record = {'p_rms', 'p_max'};

% reshape the returned rms and max fields to their original position
sensor_data.p_rms = reshape(sensor_data.p_rms, [Nx, Ny]);
sensor_data.p_max = reshape(sensor_data.p_max, [Nx, Ny]);

% reshape the sensor data to its original position so that it can be
% indexed as sensor_data(x, j, t)
sensor_data = reshape(sensor_data, [Nx, Ny, kgrid.Nt]);

% compute the amplitude spectrum
[freq, amp_spect] = spect(sensor_data, 1/kgrid.dt, 'Dim', 3);

% compute the index at which the source frequency and its harmonics occur
[f1_value, f1_index] = findClosest(freq, tone_burst_freq);
[f2_value, f2_index] = findClosest(freq, 2 * tone_burst_freq);

% extract the amplitude at the source frequency and store
beam_pattern_f1 = amp_spect(:, :, f1_index);

% extract the amplitude at the second harmonic and store
beam_pattern_f2 = amp_spect(:, :, f2_index);       

% extract the integral of the total amplitude spectrum
beam_pattern_total = sum(amp_spect, 3);


