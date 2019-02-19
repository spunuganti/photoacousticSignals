% size of the computational grid
Nx = 64;    % number of grid points in the x (row) direction
x = 1e-3;   % size of the domain in the x direction [m]
dx = x/Nx;  % grid point spacing in the x direction [m]

% define the properties of the propagation medium
medium.sound_speed = 1500;      % [m/s]

% size of the initial pressure distribution
source_radius = 5;              % [grid points]

% distance between the centre of the source and the sensor
source_sensor_distance = 10;    % [grid points]

% time array
dt = 2e-9;                      % [s]
t_end = 300e-9;                 % [s]

% computation settings
input_args = {'DataCast', 'single'};


%% 1D

% create the computational grid
kgrid = kWaveGrid(Nx, dx);

% create the time array
kgrid.setTime(round(t_end / dt) + 1, dt);

% create initial pressure distribution
source.p0 = zeros(Nx, 1);
source.p0(Nx/2 - source_radius:Nx/2 + source_radius) = 1;

% define a single sensor point
sensor.mask = zeros(Nx, 1);
sensor.mask(Nx/2 + source_sensor_distance) = 1;	

% run the simulation
sensor_data_1D = kspaceFirstOrder1D(kgrid, medium, source, sensor, input_args{:});


%% 2D
% create the computational grid
kgrid = kWaveGrid(Nx, dx, Nx, dx);

% create initial pressure distribution
source.p0 = makeDisc(Nx, Nx, Nx/2, Nx/2, source_radius);

% define a single sensor point
sensor.mask = zeros(Nx, Nx);
sensor.mask(Nx/2 - source_sensor_distance, Nx/2) = 1;

% run the simulation
sensor_data_2D = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

%% 3D

% create the computational grid
kgrid = kWaveGrid(Nx, dx, Nx, dx, Nx, dx);

% create initial pressure distribution
source.p0 = makeBall(Nx, Nx, Nx, Nx/2, Nx/2, Nx/2, source_radius);

% define a single sensor point
sensor.mask = zeros(Nx, Nx, Nx);
sensor.mask(Nx/2 - source_sensor_distance, Nx/2, Nx/2) = 1;

% run the simulation
sensor_data_3D = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});
