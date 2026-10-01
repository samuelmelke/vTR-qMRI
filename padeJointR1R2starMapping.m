classdef padeJointR1R2starMapping
% padeJointR1R2starMapping  Joint R1, R2* and M0 mapping from multi-echo spoiled 
% gradient-echo (GRE) data acquired with several flip angles and, optionally,
% several repetition times. Runs on the CPU (no GPU needed), typically within
% a few seconds for a whole-brain volume.
%
% -------------------------------------------------------------------------
% How it works
% -------------------------------------------------------------------------
% Each voxel is described by the spoiled GRE signal equation
%
%     S = M0 * sin(a) * (1 - E1) / (1 - cos(a)*E1) * exp(-R2* * TE)
%
%     with  E1 = exp(-TR * R1)   and   a = B1 * nominal flip angle
%
% and three unknowns: M0, R1 and R2*. The method has two stages.
%
% Step 1, a closed-form (linear) estimate.
%   Replacing E1 by its [1/1] Pade approximant,
%       exp(-TR*R1)  ~  (1 - TR*R1/2) / (1 + TR*R1/2),
%   turns the signal equation into a straight line for every echo time:
%
%       y = -R1 * x + R1 * M0 * exp(-R2* * TE)
%
%       with  x = S * (1 + cos a) / (2 * sin a)
%             y = S * (1 - cos a) / (TR * sin a)
%
%   At a given echo time, all acquisitions in a voxel have decayed by the
%   same factor exp(-R2* TE), and this factor only appears in the
%   intercept. Grouping the measurements by echo time and subtracting the 
%   group mean,removes this shared intercept. One slope is then fitted through 
%   the mean-subtracted points of all echo times together, which gives R1.
%   The intercepts, one per echo time, decay exponentially with TE and give
%    R2* and M0.
%
% Step 2, Gauss-Newton refinement (optional, on by default).
%   The linear estimate is used as the starting point for a few Gauss-Newton
%   steps on the EXACT signal equation (no approximation). This removes
%   the small bias of the approximation and uses every sample, including echo
%   times that are not shared between acquisitions.
%
% -------------------------------------------------------------------------
% USAGE
% -------------------------------------------------------------------------
%   obj = padeJointR1R2starMapping(te, tr, fa);
%   out = obj.estimate(data, mask, b1);
%
% Protocol (one entry per acquisition, i.e. per flip angle):
%   te   : echo times in seconds. Either a cell array, te{k} = echo times of
%          acquisition k, or one vector if all acquisitions use the same TEs.
%   tr   : repetition time(s) in seconds. One value per acquisition, or a
%          single value if all acquisitions use the same TR.
%   fa   : nominal flip angles in degrees, one per acquisition.
%
% Data:
%   data : magnitude images, [nx, ny, nz, nMeasurements], with all echoes of
%          acquisition 1 first, then all echoes of acquisition 2, and so on:
%          [acq1 TE1..TEn, acq2 TE1..TEn, ...]. A 5D array [nx,ny,nz,nTE,nFA]
%          is also accepted when all acquisitions share the same echo times.
%   mask : [nx, ny, nz] logical, voxels to fit. Use [] to fit every voxel.
%   b1   : [nx, ny, nz], ratio of actual to nominal flip angle.
%          Use [] to assume B1 = 1 everywhere .
%
% Output (maps are [nx, ny, nz], NaN outside the mask):
%   out.R1      : longitudinal relaxation rate, 1/s
%   out.T1      : 1/R1, s
%   out.R2star  : effective transverse relaxation rate, 1/s
%   out.M0      : proton-density-weighted signal, same units as the data
%   out.mask    : the voxels that were actually fitted
%
% Options (set on the object before calling estimate):
%   obj.nNewton = 1   number of Gauss-Newton steps (default 1).
%                     Set to 0 to get the pure closed-form estimate.
%
% -------------------------------------------------------------------------
% Example (synthetic data, runs as-is)
% -------------------------------------------------------------------------
%   te = {(2.0 + 3.3*(0:1))*1e-3, ...     % TR 11 ms: 2 echoes
%         (2.0 + 3.3*(0:6))*1e-3, ...     % TR 28 ms: 7 echoes
%         (2.0 + 3.3*(0:11))*1e-3};       % TR 44 ms: 12 echoes
%   tr = [11 28 44]*1e-3;
%   fa = [5 15 70];
%   R1 = 1; R2star = 20; M0 = 1000;                         % ground truth
%   data = [];
%   for k = 1:3
%       E1 = exp(-tr(k)*R1);
%       S  = M0*sind(fa(k))*(1-E1)/(1-cosd(fa(k))*E1) * exp(-R2star*te{k});
%       data = cat(4, data, reshape(S,1,1,1,[]));
%   end
%   obj = padeJointR1R2starMapping(te, tr, fa);
%   out = obj.estimate(data, [], []);
%   disp([out.R1 out.R2star out.M0])                        % ~ [1 20 1000]
%
% -------------------------------------------------------------------------
% Assumptions AND Limitations
% -------------------------------------------------------------------------
% - Single compartment per voxel, perfect RF spoiling, steady state reached.
% - The closed-form stage needs at least two acquisitions that share some
%   echo times (it compares acquisitions at the same TE);  Samples at an echo
%   time no other acquisition uses do not inform R1 in step 1, but they are
%   used in step 2.
%
% -------------------------------------------------------------------------
% The interface follows gpuJointR1R2starMapping of the GACELLE toolbox
% (https://github.com/kschan0214/gacelle), so either can be used interchangeably.

% Samuel Melke Gebremedhin, samuel.gebremedhin@donders.ru.nl
% Donders Centre for Cognitive Neuroimaging, Radboud University]
% Date last modified: 1 October 2026


    properties
        te              % {nAcq x 1} cell, echo times of each acquisition (s)
        tr              % [nAcq x 1] repetition time of each acquisition (s)
        fa              % [nAcq x 1] nominal flip angle of each acquisition (deg)
        nNewton = 1     % number of Gauss-Newton refinement steps (0 = closed-form only)
    end

    methods

        function obj = padeJointR1R2starMapping(te, tr, fa)
            % Store the protocol, one entry per acquisition

            nAcq = numel(fa);

            % Allow shortcuts: one TE vector or one TR shared by all acquisitions
            if ~iscell(te)
                te = repmat({te}, nAcq, 1);
            end
            if isscalar(tr)
                tr = repmat(tr, nAcq, 1);
            end

            if numel(te) ~= nAcq || numel(tr) ~= nAcq
                error('te, tr and fa must have one entry per acquisition (got %d, %d and %d).', ...
                      numel(te), numel(tr), nAcq);
            end
            if nAcq < 2
                error('At least two acquisitions (flip angles) are needed to estimate R1.');
            end

            obj.te = cellfun(@(t) double(t(:)'), te(:), 'UniformOutput', false);   % row vectors
            obj.tr = double(tr(:));
            obj.fa = double(fa(:));

            % The closed-form stage compares acquisitions at the same echo time.
            % If every echo time in the protocol is unique, nothing can be compared.
            allTE = round([obj.te{:}] * 1e6) / 1e6;
            if numel(unique(allTE)) == numel(allTE)
                error(['No echo time is shared between acquisitions. This method needs at least ' ...
                       'two acquisitions recorded at some of the same echo times.']);
            end
        end


        function out = estimate(obj, data, mask, b1)
            % Fit R1, R2* and M0 in every voxel of the mask

            % ---------- 1. Check and organise the inputs ----------

            % A 5D array [nx,ny,nz,nTE,nFA] becomes 4D [nx,ny,nz,nTE*nFA].
            % This is valid because MATLAB stores the TE index before the FA index.
            if ndims(data) == 5
                data = reshape(data, size(data,1), size(data,2), size(data,3), []);
            end

            imageSize     = [size(data,1) size(data,2) size(data,3)];
            nMeasurements = size(data, 4);
            if nMeasurements ~= sum(cellfun(@numel, obj.te))
                error('The data have %d measurements, but the protocol lists %d echoes in total.', ...
                      nMeasurements, sum(cellfun(@numel, obj.te)));
            end

            if isempty(b1)
                b1 = ones(imageSize);
            end

            % Only fit voxels with a valid, positive signal at every echo
            data  = abs(double(data));
            valid = all(isfinite(data), 4) & all(data > 0, 4);
            if isempty(mask)
                mask = valid;
            else
                mask = logical(mask) & valid;
            end

            % ---------- 2. Describe every measurement by its own TE, TR and flip angle ----------
            % After this, measurement m (column m of the data) was acquired with
            % echo time teAll(m), repetition time trAll(m) and flip angle faAll(m).
            teAll = [];  trAll = [];  faAll = [];
            for k = 1:numel(obj.fa)
                nEchoes = numel(obj.te{k});
                teAll   = [teAll, obj.te{k}];                  %#ok<AGROW>
                trAll   = [trAll, repmat(obj.tr(k), 1, nEchoes)];  %#ok<AGROW>
                faAll   = [faAll, repmat(obj.fa(k), 1, nEchoes)];  %#ok<AGROW>
            end

            % ---------- 3. Collect the masked voxels in a 2D table ----------
            % signal is [nVoxels x nMeasurements]: one row per voxel.
            signal = reshape(data, [], nMeasurements);
            signal = signal(mask(:), :);

            % Actual flip angle per voxel and measurement, B1-corrected (degrees)
            alpha = b1(mask(:)) .* faAll;
            sinA  = sind(alpha);
            cosA  = cosd(alpha);

            % ---------- 4. Stage 1: closed-form estimate ----------
            [R1, R2star, M0] = obj.closed_form_estimate(signal, sinA, cosA, teAll, trAll);

            % ---------- 5. Stage 2: Gauss-Newton refinement on the exact equation ----------
            if obj.nNewton > 0
                [R1, R2star, M0] = obj.replace_implausible_starting_values(R1, R2star, M0, signal);
                for iteration = 1:obj.nNewton
                    [R1, R2star, M0] = obj.gauss_newton_step(signal, sinA, cosA, teAll, trAll, R1, R2star, M0);
                end
            end

            % ---------- 6. Put the results back into 3D maps ----------
            notPhysical = ~isfinite(R1) | R1 <= 0 | ~isfinite(R2star) | R2star < 0 | ~isfinite(M0) | M0 <= 0;
            R1(notPhysical) = NaN;  R2star(notPhysical) = NaN;  M0(notPhysical) = NaN;

            out.R1     = obj.to_map(R1,     mask, imageSize);
            out.T1     = 1 ./ out.R1;
            out.R2star = obj.to_map(R2star, mask, imageSize);
            out.M0     = obj.to_map(M0,     mask, imageSize);
            out.mask   = mask;
        end

    end


    methods (Static, Access = private)

        function [R1, R2star, M0] = closed_form_estimate(signal, sinA, cosA, teAll, trAll)
            % Stage 1: the linear Pade estimate (see the help text at the top)

            % Transform the signal so that R1 becomes the slope of a straight line
            x = signal .* (1 + cosA) ./ (2 .* sinA);
            y = signal .* (1 - cosA) ./ (trAll .* sinA);

            % Group the measurements by echo time. Rounding to a microsecond
            % makes equal echo times from different acquisitions match exactly.
            teRounded = round(teAll * 1e6) / 1e6;
            uniqueTE  = unique(teRounded);
            nVoxels   = size(signal, 1);
            nGroups   = numel(uniqueTE);

            % Within each echo-time group, subtract the group mean. This removes
            % the shared intercept R1*M0*exp(-R2* TE) and leaves only the slope.
            xMean = zeros(nVoxels, nGroups);  yMean = zeros(nVoxels, nGroups);
            xCentred = zeros(size(x));        yCentred = zeros(size(y));
            for g = 1:nGroups
                inGroup           = (teRounded == uniqueTE(g));
                xMean(:,g)        = mean(x(:,inGroup), 2);
                yMean(:,g)        = mean(y(:,inGroup), 2);
                xCentred(:,inGroup) = x(:,inGroup) - xMean(:,g);
                yCentred(:,inGroup) = y(:,inGroup) - yMean(:,g);
            end

            % The common slope of y versus x is -R1 (least squares, all groups together)
            slope = sum(xCentred .* yCentred, 2) ./ sum(xCentred.^2, 2);
            R1    = -slope;

            % The intercept of each echo-time group equals R1*M0*exp(-R2* TE)
            intercept = yMean - slope .* xMean;       % [nVoxels x nGroups]

            % A straight-line fit of log(intercept) against TE then gives
            %   log(intercept) = log(R1*M0) - R2* * TE
            % Only positive intercepts can be log-transformed; the others are ignored.
            use         = intercept > 0;
            logInt      = log(max(intercept, realmin));
            nUsed       = sum(use, 2);
            teMean      = sum(use .* uniqueTE, 2) ./ nUsed;
            logIntMean  = sum(use .* logInt,   2) ./ nUsed;
            teDiff      = uniqueTE - teMean;
            slopeTE     = sum(use .* teDiff .* (logInt - logIntMean), 2) ./ sum(use .* teDiff.^2, 2);

            R2star = -slopeTE;
            M0     = exp(logIntMean - slopeTE .* teMean) ./ R1;
        end


        function [R1, R2star, M0] = gauss_newton_step(signal, sinA, cosA, teAll, trAll, R1, R2star, M0)
            % Stage 2: one Gauss-Newton step on the exact signal equation

            % Model signal with the current estimates
            E1    = exp(-trAll .* R1);
            denom = 1 - cosA .* E1;
            decay = exp(-R2star .* teAll);
            model = M0 .* sinA .* (1 - E1) ./ denom .* decay;
            residual = model - signal;

            % Derivatives of the model with respect to each unknown
            dM0     = model ./ M0;
            dR1     = M0 .* sinA .* trAll .* E1 .* (1 - cosA) ./ denom.^2 .* decay;
            dR2star = -teAll .* model;

            % Normal equations (J'J) * step = -J' * residual, one 3x3 system per voxel.
            % J'J is symmetric, so only six of its nine entries are needed.
            A11 = sum(dM0.^2, 2);       A12 = sum(dM0 .* dR1, 2);    A13 = sum(dM0 .* dR2star, 2);
            A22 = sum(dR1.^2, 2);       A23 = sum(dR1 .* dR2star, 2);
            A33 = sum(dR2star.^2, 2);
            g1  = -sum(dM0 .* residual, 2);
            g2  = -sum(dR1 .* residual, 2);
            g3  = -sum(dR2star .* residual, 2);

            [stepM0, stepR1, stepR2star] = padeJointR1R2starMapping.solve_symmetric_3x3( ...
                A11, A12, A13, A22, A23, A33, g1, g2, g3);

            M0     = M0     + stepM0;
            R1     = R1     + stepR1;
            R2star = R2star + stepR2star;
        end


        function [d1, d2, d3] = solve_symmetric_3x3(A11, A12, A13, A22, A23, A33, g1, g2, g3)
            % Solve A*d = g for many symmetric 3x3 systems at once (one per voxel),
            % using the inverse written out explicitly (cofactors / determinant).
            % Voxels with a singular system get a zero step, i.e. are left unchanged.
            C11 = A22.*A33 - A23.^2;
            C12 = A13.*A23 - A12.*A33;
            C13 = A12.*A23 - A13.*A22;
            C22 = A11.*A33 - A13.^2;
            C23 = A12.*A13 - A11.*A23;
            C33 = A11.*A22 - A12.^2;
            determinant = A11.*C11 + A12.*C12 + A13.*C13;

            d1 = (C11.*g1 + C12.*g2 + C13.*g3) ./ determinant;
            d2 = (C12.*g1 + C22.*g2 + C23.*g3) ./ determinant;
            d3 = (C13.*g1 + C23.*g2 + C33.*g3) ./ determinant;

            singular = ~isfinite(determinant) | abs(determinant) < eps;
            d1(singular) = 0;  d2(singular) = 0;  d3(singular) = 0;
        end


        function [R1, R2star, M0] = replace_implausible_starting_values(R1, R2star, M0, signal)
            % Give Gauss-Newton a reasonable starting point where stage 1 failed
            bad = ~isfinite(R1) | R1 <= 0 | R1 > 10;          R1(bad)     = 1;    % 1/s
            bad = ~isfinite(R2star) | R2star < 0 | R2star > 400; R2star(bad) = 30;   % 1/s
            bad = ~isfinite(M0) | M0 <= 0;                    M0(bad)     = signal(bad, 1);
        end


        function map = to_map(values, mask, imageSize)
            % Put the per-voxel values back into a 3D image, NaN outside the mask
            map       = nan(imageSize);
            map(mask) = values;
        end

    end
end