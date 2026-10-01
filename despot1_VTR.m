classdef despot1_VTR
% Very fast Variable-TR DESPOT1: R1 and M0 from spoiled GRE amplitudes acquired with
% different flip angles and different repetition times.
%
% Model (Deoni et al. MRM 2003;49(3):515-526):
%   S = M0 sin(a) (1-E1) / (1 - E1 cos(a)),  E1 = exp(-TR R1)
%
% The standard DESPOT1 linearisation assumes one TR shared by all flip
% angles, so that E1 is a single constant to be regressed. With variable TR
% that no longer holds. Here E1 is instead eliminated by approximating
%   1 - exp(-TR R1)  ~  TR R1 / (1 + TR R1 / 2)     ([1/1] Pade approximant)
% which keeps the system linear in the unknowns x = [R1 ; R1*M0]:
%   z = S (1-cos a) / (TR sin a)        (per acquisition)
%   A = [ -S (1+cos a) / (2 sin a) , 1 ]
%   z = A x,  solved by least squares over all acquisitions.
%
% Because the system is only 2x2, the normal equations are solved in closed
% form for every voxel at once (sums taken over the acquisition dimension),
% so the whole volume is fitted in a few vectorised operations with no
% per-voxel loop.
%
% estimate()        : linear Pade solution, vectorised over all voxels.
% refine_newton()   : optional Gauss-Newton steps against the exact model,
%                     which removes the linearisation bias.
% estimate_newton() : estimate() followed by refine_newton().
%
% Derived from despot1.m of the GACELLE toolbox
% (https://github.com/kschan0214/gacelle):
%   
% Variable-TR version:
%   Samuel Melke Gebremedhin, samuel.gebremedhin@donders.ru.nl
%   Donders Centre for Cognitive Neuroimaging, Radboud University
%   Date last modified: 2 September 2026
%


    properties (Constant)
        gyro = 42.57747892;
    end
    properties (GetAccess = public, SetAccess = protected)
        tr;     % [nSeq x 1] repetition times
        fa;     % [nSeq x 1] flip angles (deg)
    end

    methods
        function obj = despot1_VTR(tr,fa)
            obj.tr = double(tr(:));
            obj.fa = double(fa(:));
        end

        %% ---------- LINEAR Pade estimate (no Newton), fully vectorised ----------
        function [t1, m0, mask_fitted] = estimate(obj,img,mask,b1)
        % img : magnitude, [x,y,z,nSeq] (per-flip amplitudes, e.g. S0 or 1st echo)
        % mask: [x,y,z] (optional) ; b1 : [x,y,z] (optional, true/nominal flip ratio)
            dims = size(img); nSeq = dims(4);
            if nargin < 4 || isempty(b1),   b1   = ones(dims(1:3)); end
            if nargin < 3 || isempty(mask), mask = ones(dims(1:3)); end

            % flatten to [nVox, nSeq] and restrict to mask
            S    = reshape(abs(double(img)), [prod(dims(1:3)) nSeq]);
            b1v  = reshape(double(b1),       [prod(dims(1:3)) 1]);
            mkv  = reshape(mask>0,           [prod(dims(1:3)) 1]);
            ind  = find(mkv);
            Sm   = S(ind,:);                                   % masked image [Nm, nSeq] 
            Bm   = b1v(ind);                                   % masked b1 [Nm, 1]

            alpha = Bm .* obj.fa(:).';                         % [Nm, nSeq] true flip (deg)
            TRr   = obj.tr(:).';                               % [1, nSeq]
            sa = sind(alpha); ca = cosd(alpha);

            % z = A x ,  x = [R1; R1*M0]  
            z = Sm .* (1 - ca) ./ (TRr .* sa);                 % [Nm, nSeq] auxiliary variable z_i
            c = -Sm .* (1 + ca) ./ (2 .* sa);                  % A_i(:,1)  (Pade column) 
            % --- 2x2 normal equations, summed over flips (dim 2) ---
            Sc2 = sum(c.^2,2);  Sc = sum(c,2);  Nf = nSeq;
            Scz = sum(c.*z,2);  Sz = sum(z,2);
            det = Sc2.*Nf - Sc.^2;  det(det==0) = eps;
            R1   = ( Nf.*Scz  - Sc.*Sz  ) ./ det;              % x(1) = R1
            R1M0 = ( Sc2.*Sz  - Sc.*Scz ) ./ det;              % x(2) = R1*M0
            T1m  = 1 ./ R1;
            M0m  = R1M0 ./ R1;

            [t1,m0,mask_fitted] = obj.scatter_and_clean(T1m,M0m,ind,dims,mask);
        end

        %% ---------- Newton (Gauss-Newton) refinement on the EXACT model ----------
        function [t1, m0, mask_fitted] = refine_newton(obj,img,mask,b1,t1_0,m0_0,niter)
        % Refine a given (t1_0,m0_0) starting estimate against the exact Ernst
        % steady-state model. niter Gauss-Newton steps (default 1).
            if nargin < 7 || isempty(niter), niter = 1; end
            dims = size(img); nSeq = dims(4);
            if nargin < 4 || isempty(b1),   b1   = ones(dims(1:3)); end
            if nargin < 3 || isempty(mask), mask = ones(dims(1:3)); end

            S   = reshape(abs(double(img)), [prod(dims(1:3)) nSeq]);
            b1v = reshape(double(b1),       [prod(dims(1:3)) 1]);
            mkv = reshape(mask>0,           [prod(dims(1:3)) 1]);
            T10 = reshape(double(t1_0),     [prod(dims(1:3)) 1]);
            M00 = reshape(double(m0_0),     [prod(dims(1:3)) 1]);
            ind = find(mkv & T10>0 & isfinite(T10) & isfinite(M00));

            Sm = S(ind,:); Bm = b1v(ind);
            alpha = Bm .* obj.fa(:).';  TRr = obj.tr(:).';
            sa = sind(alpha); ca = cosd(alpha);
            t1 = T10(ind);  m0 = M00(ind);                     % [Nm,1]

            for it = 1:niter
                E1  = exp(-TRr ./ t1);                          % [Nm,nSeq]
                den = 1 - ca.*E1;
                G   = sa.*(1 - E1)./den;                        % ds/dM0
                f   = m0 .* G;                                  % model
                r   = f - Sm;                                   % residual
                % ds/dT1 = m0 sin a (cos a -1) E1 TR / (T1^2 (1-cos a E1)^2)
                dsdT1 = m0 .* sa .* (ca - 1) .* E1 .* TRr ./ (t1.^2 .* den.^2);
                % 2x2 Gauss-Newton normal equations (sum over flips)
                A11 = sum(G.^2,2);     A12 = sum(G.*dsdT1,2);   A22 = sum(dsdT1.^2,2);
                g1  = sum(G.*r,2);     g2  = sum(dsdT1.*r,2);
                dj  = A11.*A22 - A12.^2;  dj(dj==0) = eps;
                dm0 = -( A22.*g1 - A12.*g2 ) ./ dj;
                dt1 = -( A11.*g2 - A12.*g1 ) ./ dj;
                m0  = m0 + dm0;  t1 = t1 + dt1;
                bad = t1<=0;  t1(bad) = 0;  m0(bad) = 0;
            end

            [t1,m0,mask_fitted] = obj.scatter_and_clean(t1,m0,ind,dims,mask);
        end

        %% ---------- convenience: linear Pade + Newton (old behaviour) ----------
        function [t1, m0, mask_fitted] = estimate_newton(obj,img,mask,b1,niter)
            if nargin < 5 || isempty(niter), niter = 1; end
            if nargin < 4, b1 = []; end
            if nargin < 3, mask = []; end
            [t1L,m0L]               = obj.estimate(img,mask,b1);
            [t1,m0,mask_fitted]     = obj.refine_newton(img,mask,b1,t1L,m0L,niter);
        end
    end

    methods (Static, Access = protected)
        function [t1,m0,mask_fitted] = scatter_and_clean(T1m,M0m,ind,dims,mask)
        % scatter masked-voxel results back to volume + drop infeasible values
            t1 = zeros(prod(dims(1:3)),1);  m0 = zeros(prod(dims(1:3)),1);
            t1(ind) = T1m;  m0(ind) = M0m;
            t1 = reshape(t1,dims(1:3));  m0 = reshape(m0,dims(1:3));
            mask_fitted = ones(size(t1));
            mask_fitted(t1<=0)     = 0;  mask_fitted(isnan(t1)) = 0;  mask_fitted(isinf(t1)) = 0;
            mask_fitted(m0<0)      = 0;  mask_fitted(isnan(m0)) = 0;  mask_fitted(isinf(m0)) = 0;
            mask_fitted = mask_fitted .* (mask>0);
            t1 = t1 .* mask_fitted;  m0 = m0 .* mask_fitted;
        end
    end
end
