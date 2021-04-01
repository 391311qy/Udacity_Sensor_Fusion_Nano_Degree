% TODO : Find the Bsweep of chirp for 1 m resolution
fb = [0, 1.1, 13, 24]; % MHz
Rmax = 300; %m = c * fb * Ts / (2 * Bsweep)
c = 3 * 10 ^ 8;%m/s
dr = 1;%m
                                                                                                                                                                                                                                                                                                                                                                                        
% TODO : Calculate the chirp time based on the Radar's Max Range
Ts = 5.5 * 2 / c;

% TODO : define the frequency shifts 
Bsweep = c / (2*dr);

calculated_range = c * fb * Ts / (2 * Bsweep);
% Display the calculated range
disp(calculated_range);

