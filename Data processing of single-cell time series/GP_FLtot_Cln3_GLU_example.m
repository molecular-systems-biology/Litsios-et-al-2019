%% data input
data_gfp = csvread('FLmean_Cln3_GLU.csv',0,0); % read mean GFP fluorescence data
data_vol = csvread('Vol_Cln3_GLU.csv',0,0);    % read cell volume data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initializations
T = data_gfp(:,1); % vector of measurement times
mu_all = {};       % Gaussian Process (GP) mean for total GFP abundance
dmu_all = {};      % GP mean for time derivative of total GFP abundance
sd_all = {};       % GP standard deviation for total GFP abundance
dmu_sd_all = {};   % GP standard deviation for time derivative of total GFP abundance
t_all = {};        % time vector of smoothed cell data
raw_t = {};        % time vector of raw cell data
raw_gfpt = {};     % raw total GFP data
Hyp_all = {};      % hyperparameter of GP fit
cellnum = size(data_gfp,2)-1; % number of cells followed

% bud appearance times (minutes) for all cells
budtimes_GLU = [81    93    84    45   138    30    60    33    36   192   114   105    45    54    39   126    60 ...
 171    42    30    30    42    90    90    75    36    63   150    63    60    45    51    54    30 ...
 45    69    66    51    96    39   111    30    99   141    39   102    63    72    30   108    27] ;

% cells that display a noisy total GFP time series require manual assignment of
% measurement noise standard deviation to avoid artifacts in GP fit
noisycells = [2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 20 21 23 24 25 26 27 28 29 31 32 33 34 38 40 42 43 44 45 46 48 49 50 51];

% log-standard deviation of measurement noise used in GP fit. "Noisy" cells
% are assigned a fixed logsigma value (i.e. logsigma prior is a delta
% distribution). For all other cells, logsigma is obtained through marginal
% likelihood optimization
logsigma = -3*ones(1,50);
logsigma(2) = 4;
logsigma(3) = 4.25;
logsigma(4) = 4;
logsigma(5) = 4.75;
logsigma(6) = 4.25;
logsigma(8) = 3.75;
logsigma(9) = 4.25;
logsigma(10) = 4.5;
logsigma(11) = 4.25;
logsigma(12) = 4.75;
logsigma(13) = 4.5;
logsigma(14) = 3.5;
logsigma(15) = 4.15;
logsigma(16) = 3.9;
logsigma(17) = 4;
logsigma(18) = 4.35;
logsigma(20) = 3.5;
logsigma(21) = 2.95;
logsigma(22) = 4.5;
logsigma(23) = 3.5;
logsigma(24) = 3.75;
logsigma(25) = 3.9;
logsigma(26) = 3.5;
logsigma(27) = 3.7;
logsigma(28) = 3.5;
logsigma(29) = 3.95;
logsigma(31) = 3.95;
logsigma(32) = 3.95;
logsigma(33) = 3.75;
logsigma(34) = 3.3;
logsigma(38) = 3.95;
logsigma(40) = 3.5;
logsigma(42) = 3.75;
logsigma(43) = 3.5;
logsigma(44) = 4.2;
logsigma(45) = 4;
logsigma(46) = 3.95;
logsigma(47) = 3.25;
logsigma(48) = 4.65;
logsigma(49) = 4.2;
logsigma(50) = 4.45;
logsigma(51) = 4.25;

% time points dropped from individual cell trajectories due to obvious
% imaging artifacts (e.g. out-of-focus images, failed tracking etc.) which
% generate "outlier" time points in the data

droppoints = cell(1,cellnum);
droppoints{16} = 11:13;
droppoints{18} = 29:44;
droppoints{22} = 1;
droppoints{29} = 28;
droppoints{31} = [10 13 14];
droppoints{33} = 10;
droppoints{34} = 16;
droppoints{38} = 16;
droppoints{42} = [8 9 18];
droppoints{49} = [7 8 10 11];

% cells with extremely distorted time series which are cannot be further analyzed
dropcells = [3 16 20 21 24 26 30 44 49 51];
allcells = 1:cellnum;
allcells(dropcells) = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GP fitting of the single-cell time series
for ncell = allcells
    fprintf('Processing cell %d\n',ncell)
    L = nnz(data_gfp(:,ncell+1));
    x = T(1:L);
    y1 = data_gfp(1:L,ncell+1).*data_vol(1:L,ncell+1); % calculate total GFP from mean GFP and volume data
    y1(droppoints{ncell}) = [];
    x(droppoints{ncell}) = [];
    y = y1-mean(y1);  % GP fitting with a zero prior mean function works better with data that has a mean of zero
    %z = linspace(x(1),x(end),100)';
    z = [x(1):x(end)]';
    % initializations for GP fit
    Hyp = struct('mean',{},'cov',{},'lik',{});
    Hyp(1).mean = 0;Hyp(1).cov = 0;Hyp1(1).lik = 0;
    for n = 1:20
        Hyp(n).mean = 0;Hyp(n).cov = 0;Hyp1(n).lik = 0;
    end
    meanfunc = [];       % prior mean is zero
    covfunc = @covRQiso; % using a rational quadratic covariance function...
    likfunc = @likGauss; % ... and Gaussian likelihood function
    % GP is fitted by minimizing the negative log-marginal likelihood of the data with respect to hyperparameters
    % To avoid local minima, optimization is run 20 times starting from
    % random initial hyperparameter values
    parfor n = 1:20 
        if ~ismember(ncell,noisycells)
            likfunc = @likGauss; 
            %random parameterization of the covariance function and the
            %likelihood standard deviation
            hyp = struct('mean', [], 'cov', [2*randn 2*randn 0], 'lik', randn); 
            % GP fitting
            hyp_GP = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x,y);
            % extract the obtained GP parameterization
            Hyp(n).mean = hyp_GP.mean;
            Hyp(n).cov = hyp_GP.cov;
            Hyp(n).lik = hyp_GP.lik;
            % store the achieved marginal likelihood value
            ML(n) = gp(Hyp(n), @infGaussLik, meanfunc, covfunc, likfunc, x, y);
        else
            prior = [];
            % if cell belongs to "noisy" cells, fix the logsigma prior to a
            % delta function (i.e. do not optimize over it) 
            prior.lik = {{@priorDelta}};
            likfunc = @likGauss;
            inf  = {@infPrior,@infGaussLik,prior}; 
            hyp = struct('mean', [], 'cov', [2*randn 2*randn 0], 'lik', logsigma(ncell)); %@covRQard
            % GP fitting
            hyp_GP = minimize(hyp, @gp, -100, inf, meanfunc, covfunc, likfunc, x,y);
            % extract the obtained GP parameterization
            Hyp(n).mean = hyp_GP.mean;
            Hyp(n).cov = hyp_GP.cov;
            Hyp(n).lik = hyp_GP.lik;
            % store the achieved marginal likelihood value
            ML(n) = gp(Hyp(n), @infGaussLik, meanfunc, covfunc, likfunc, x, y);
        end
    end
    %% Making predictions using the optimal hyperparameters
    [minML,pos] = min(ML); % locate the smallest value of the negative log-marginal likelihood
    % and make predictions for the posterior process with the optimal hyperparameters 
    % (at time points specified by z) using the single cell measurements
    [mu s2] = gp(Hyp(pos), @infGaussLik, meanfunc, covfunc, likfunc, x, y, z);
    mu = mu+mean(y1); % add back the data mean to the posterior mean function
    % extract posterior GP mean and standard deviation; store single-cell
    % measurements and optimal hyperparameters
    mu_all{ncell} = mu;  % smoothed total GFP abundance of a single cell
    sd_all{ncell} = s2;  % standard deviation of total GFP abundance of a single cell
    t_all{ncell} = z;
    raw_t{ncell} = x;
    raw_gfpt{ncell} = y1;
    Hyp_all{ncell} = Hyp(pos);
    %% GP derivative calculation (using the formulas of Swain et al. 2016)
    % (calculation requires first and second partial derivatives of the
    % covariance function, so the optimal covariance function parameters are needed here)
    
    sigma_f = exp(Hyp(pos).cov(2));
    l = exp(Hyp(pos).cov(1));
    alpha = exp(Hyp(pos).cov(3));
    sigma_noise = exp(Hyp(pos).lik);   
    K = zeros(length(x));
    for m = 1:length(x)
        for n = 1:length(x)
            K(m,n) = sigma_f^2*(1+1/(2*alpha*l^2)*(x(m)-x(n))^2)^-alpha;
        end
    end
    
    Kpost = zeros(length(z),length(x));
    for m = 1:length(z)
        for n = 1:length(x)
            Kpost(m,n) = sigma_f^2*(1+1/(2*alpha*l^2)*(z(m)-x(n))^2)^-alpha;
        end
    end
    
    ypost = Kpost*((K+(sigma_noise^2)*eye(size(K)))\y);
    
    d1Kpost = zeros(length(z),length(x));
    for m = 1:length(z)
        for n = 1:length(x)
            d1Kpost(m,n) = -alpha*sigma_f^2*(1+1/(2*alpha*l^2)*(z(m)-x(n))^2)^(-alpha-1)*1/(alpha*l^2)*(z(m)-x(n));
        end
    end
    
    dypost = d1Kpost*((K+(sigma_noise^2)*eye(size(K)))\y);
    
    d1d2Kpost = zeros(length(z),length(z));
    for m = 1:length(z)
        for n = 1:length(z)
            d1d2Kpost(m,n) = alpha*(alpha+1)*sigma_f^2*(1+1/(2*alpha*l^2)*(z(m)-z(n))^2)^(-alpha-2)*(-1/(alpha*l^2)^2*(z(n)-z(m))^2)+...
                alpha*sigma_f^2*(1+1/(2*alpha*l^2)*(z(m)-z(n))^2)^(-alpha-1)*1/(alpha*l^2);
        end
    end
 
    covdypost = d1d2Kpost - d1Kpost*inv((K+(sigma_noise^2)*eye(size(K))))*d1Kpost';
    vardypost = diag(covdypost);
    
    dmu_all{ncell} = dypost;  % derivative of smoothed total GFP abundance of a single cell
    dmu_sd_all{ncell} = sqrt(vardypost); % standard deviation of derivative of smoothed total GFP abundance
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Store data obtained by GP fitting

%save('Cln3_GLU_smooth.mat','raw_t','raw_gfpt','mu_all','sd_all','t_all','Hyp_all','dmu_all','dmu_sd_all')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Post-process data to align trajectories in normalized time
% (Note: each cell is followed for several time points (typically 6) after
% budding. This is done to avoid edge-related artifacts in the GP fit.
% The extra time points now need to be removed using the information on the
% bud appearance time for each cell)
gfp_normtime = zeros(length(mu_all),100);
cln3_normtime = zeros(length(mu_all),100);
bud_p = [];
% in normalized time, each single-cell trajectory is linearly interpolated
% at 100 time points (from time zero until budding)
for n = allcells
    bud_p(n) = find(budtimes_GLU(n)<=t_all{n},1);
    % total GFP for each cell
    gfp_normtime(n,:) = interp1(t_all{n}(1:bud_p(n))',max(0,mu_all{n}(1:bud_p(n))'),linspace(0,t_all{n}(bud_p(n)),100));
    % Cln3 abundance (GFP derivative) for each cell
    cln3_normtime(n,:) = interp1(t_all{n}(1:bud_p(n))',max(0,dmu_all{n}(1:bud_p(n))'),linspace(0,t_all{n}(bud_p(n)),100));
end
cln3_normtime = cln3_normtime(allcells,:);
gfp_normtime = gfp_normtime(allcells,:);
t_axis_normtime = 0:99;
% WARNING: data matrices may contain NaNs. Use nanmean() to compute
% averages correctly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %% Post-process data to align trajectories in real time
% (Note: each cell is followed for several time points (typically 6) after
% budding. This is done to avoid edge-related artifacts in the GP fit.
% The extra time points now need to be removed using the information on the
% bud appearance time for each cell)
T_over = 18;  % cells are followed for at most 18 minutes after budding
endtimes = [];
for n = allcells
    endtimes(n) = t_all{n}(end)-T_over;
end
tmax = max(endtimes);
tmin = min(endtimes);
cln3_realtime = zeros(length(mu_all),tmax+1);
gfp_realtime = zeros(length(mu_all),tmax+1);
% in real time, all single-cell trajectories are aligned at their endpoint
% (corresponding to bud appearance). The time step for all time series is
% the same (1 minute)
for n = allcells
    % Cln3 abundance (GFP derivative) for each cell
    cln3_realtime(n,tmax-budtimes_GLU(n)+1:tmax+1) = interp1(t_all{n}(1:end)',max(0,dmu_all{n}'),0:budtimes_GLU(n));
    cln3_realtime(n,cln3_realtime(n,:)==0) = NaN;
    % total GFP for each cell
    gfp_realtime(n,tmax-budtimes_GLU(n)+1:tmax+1) = interp1(t_all{n}(1:end)',max(0,mu_all{n}'),0:budtimes_GLU(n));
    gfp_realtime(n,gfp_realtime(n,:)==0) = NaN;
end
cln3_realtime = cln3_realtime(allcells,:);
gfp_realtime = gfp_realtime(allcells,:);
t_axis_realtime = -tmax:0;
% WARNING: data matrices will contain NaNs. Use nanmean() to compute
% averages correctly


