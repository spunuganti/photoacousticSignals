% Photoacoustic Waveforms in 1D, 2D and 3D Example
%
% The time-varying pressure signals recorded from a photoacoustic source
% look different depending on the number of dimensions used in the
% simulation. This difference occurs because a point source in 1D
% corresponds to a plane wave in 3D, and a point source in 2D corresponds
% to an infinite line source in 3D. This examples shows the difference
% between the signals recorded in each dimension. It builds on the
% Simulations in One Dimension, Homogeneous Propagation Medium, and
% Simulations in Three Dimensions examples.
%
% author: Bradley Treeby and Ben Cox
% date: 29th January 2011
% last update: 20th June 2017
%  
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2011-2017 Bradley Treeby and Ben Cox

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

clearvars;

% =========================================================================
% SETTINGS
% =========================================================================

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


% define the properties of the propagation medium
medium.sound_speed = 1500;      % [m/s]

% size of the initial pressure distribution
source_radius = 0.01;              % [grid points]

% distance between the centre of the source and the sensor
source_sensor_distance = 10;    % [grid points]

% time array
dt = 25e-9;                      % [s]
t_end = 5e-5;                 % [s] 2000 samples

% computation settings
input_args = {'DataCast', 'single'};

% =========================================================================
% THREE DIMENSIONAL SIMULATION
% =========================================================================

% create the computational grid
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);

% create the time array
kgrid.setTime(round(t_end / dt) + 1, dt);
% 
% % create initial pressure distribution
% source.p0 = makeBall(Nx, Ny, Nz, Nx/2, Ny/2, Nz/2, source_radius)+makeSphere(Nx, Ny, Nz, 1);


% define grid parameters
grid_size           = [Nx, Ny, Nz];

% create a Cartesian sphere with the x, y, z positions of the bowls
sphere_radius       = 20e-3;
num_bowls           = 64;
bowl_pos            = makeCartSphere(sphere_radius, num_bowls, [x/2,y/2,z/2]).';

% convert the Cartesian bowl positions to grid points
bowl_pos            = round(bowl_pos/dx);

% define element parameters
radius              = round(x/ (2 * dx));
diameter            = 1;
focus_pos           = [Nx/2,Ny/2,Nz/2];

% create bowls
source.p0=makeMultiBowl(grid_size, bowl_pos, radius, diameter, focus_pos, 'Plot', true);

% define a single sensor point
sensor.mask = zeros(Nx, Ny, Nz);
sensor.mask(Nx/2 - source_sensor_distance, Ny/2, Nz/2) = 1;

% run the simulation
sensor_data_3D = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});

% =========================================================================
% VISUALISATION
% =========================================================================

% plot the time signals recorded in each dimension
figure;
[t_sc, t_scale, t_prefix] = scaleSI(t_end);
%plot(kgrid.t_array * t_scale, sensor_data_1D ./ max(abs(sensor_data_1D)), 'b-');
hold on;
%plot(kgrid.t_array * t_scale, sensor_data_2D ./ max(abs(sensor_data_2D)), 'r-');
plot(kgrid.t_array * t_scale, sensor_data_3D ./ max(abs(sensor_data_3D)), 'k-');
xlabel(['Time [' t_prefix 's]']);
ylabel('Recorded Pressure [au]');
%legend('1D', '2D', '3D');
axis tight;