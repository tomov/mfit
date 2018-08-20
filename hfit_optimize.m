function results = hfit_optimize(likfun,hparam,param,data)

    % Estimate hyperparameters (parameters for a group-level prior) by inverting
    % the generative model:
    %   group-level hyperparameters h ~ P(h), defined by hparam.logpdf
    %   individual parameters       x ~ P(x|h), defined by param.hlogpdf (for any h)
    %                                           and param.logpdf (for a fixed h)
    %   data                     data ~ P(data|x), defined by likfun 
    %
    % Uses expectation maximization (EM) to find MAP P(h|data), with x serving as latent
    % variables (Bishop 2006). We approximate the integral over x using importance
    % sampling, with x's resampled every few iterations using Gibbs sampling, 
    % using Metropolis-Hastings to sample each component.
    %
    % USAGE: param = hfit_optimize(likfun,hparam,param,data)
    %        results = mfit_optimize(likfun,param,data,[nstarts])
    %
    % INPUTS:
    %   likfun - likelihood function handle
    %   hparam - [K x 1] hyperparameter structure
    %   param - [K x 1] parameter structure, with hlogpdf and hrnd fields
    %   data - [S x 1] data structure
    %
    % OUTPUTS:
    %   param - the param structure array with .logpdf defined based on
    %           the MAP hyperparameters h. Can be passed in turn to mfit_optimize.
    %
    % Sam Gershman, July 2017

    
    % initialization
    tol = 1e-3;
    maxiter = 20;
    nsamples = 10;
    iter = 0;
    K = length(param);
    assert(length(hparam) == K, 'param and hparam must have the same length');
    S = length(data);
%    m = randn(1,K);
%    v = ones(1,K)*100;

    h_old = [];
    for k = 1:K
        h_old = [h_old hparam(k).rnd()];
    end
    
    % extract lower and upper bounds
    lb = [hparam.lb];
    ub = [hparam.ub];
    
    options = optimset('Display','off');
    warning off all
    
    % run expectation-maximization
    while iter < maxiter
        
        iter = iter + 1;
        disp(['.. iteration ',num2str(iter)]);
  
        % Draw samples from P(x|data,h_old)
        % TODO only do every so often
        [X, logq] = samples(h_old, likfun, hparam, param, data, nsamples);

        % compute P(x|data,h_old) for samples
        logp = mfit_post(X, param, data, likfun);
        logp = logp - logsumexp(logp); % normalize by P(data,h_old) (approximately)

        % compute importance weights
        w = exp(logp - logq);

        Q = @(h_new) -computeQ(h_new, h_old, X, w, data, hparam, param, likfun);

        h0 = h_old;
        % h0 = [];  % TODO test
        % for k = 1:K
        %     h0 = [h_old hparam(k).rnd()];
        % end
    
        [h_new,nlogp] = fmincon(f,h0,[],[],[],[],lb,ub,[],options);
        logp = -nlogp;
            
        h_old = h_new;
    end

    param = set_logpdf(hparam, param, h);
end

% Draw random samples of paramters from P(x|data,h) using Gibbs sampling.
% Notice that q(x) is accurate only up to a proportionality constant,
% but that's okay because we always P()
%
% OUTPUTS:
%   X = [nsamples x K] samples; each row is a set of parameters x
%   logq = [nsamples x 1] = ln q(x) = ln P(x|data,h) for each x; used to compute importance weights

%
function [X, logq] = sample(h_old, likfun, hparam, param, data, nsamples)
    K = length(param);
    X = nan(nsamples, K);
    logq = nan(nsamples, 1);
    batch_size = 50; % subsample once every batch_size samples

    % initialize x0
    x_old = param_rnd(hparam, param, h_old);

    % set .logpdf = P(x|h)
    param = set_logpdf(hparam, param, h);

    % log of std of proposal distribution for each component
    ls = zeros(1,K);

    % Draw nsamples samples
    for n = 1:nsamples % TODO burn-in
        x_new = nan(1,K);

        % Adaptive Metropolis-within-Gibbs for each component
        % See Roberts and Rosenthal (2008): Examples of Adaptive MCMC
        for k = 1:K
            % P(x_k|x_\k,data,h) is proportional to P(data|x) P(x_k|h)
            logpost_k = @(x_k) likfun([x_new(1:k-1) x_k x_old(k+1:end)], data) + param(k).logpdf(x_k);

            % proposals update by adding an increment ~ N(0,exp(ls(k))
            proprnd = @(x_old) normrnd(x_old, exp(ls(k));
            proppdf = @(x_new,x_old) normpdf(x_new, x_old, exp(ls(k));

            % draw batch_size samples of k-th component and take the last one only
            [x, accept] = mhsample(x_old(k), batch_size, 'logpdf', post_k, 'proprnd', proprnd, 'symmetric', true) 
            x_new(k) = x(batch_size); % ignore first batch_size-1 samples

            % update proposal distribution to maintain the acceptance rate
            % around 0.44
            d = min(0.01, sqrt(n));
            if accept > 0.44
                ls(k) = ls(k) + d;
            else
                ls(k) = ls(k) - d;
            end
        end

        X(n,:) = x_new;
        x_old = x_new;
    end

    % compute q(x) = P(x|data,h) up to a proportionality constant
    logq = mfit_post(X, param, data, likfun);

    % Compute proportionality constant p(data,h) and normalize q(x)'s by it.
    % This is important since the normalization constant is different for different
    % values of h, and thus ignoring it would screw up the importance weights.
    C = logsumexp(logq);
    logq = logq - C;
end

% random draw from P(x|h)
%
function x = param_rnd(hparam, param, h)
    i = 1;
    for k = 1:length(param)
        l = length(hparam(k).lb);
        while true
            x(k) = param(k).hrnd(h(i:i+l-1));
            if param(k).lb <= x(k) and x(k) >= param(k).ub
                break;
            end
        end
        i = i + l;
    end
end

% set .logpdf i.e. P(x|h) for each param based on given hyperparameters h
%
function param = set_logpdf(hparam, param, h)
    i = 1;
    for k = 1:length(param)
        l = length(hparam(k).lb);
        param(k).logpdf = @(x) param(k).hlogdf(x,h(i:i+l-1));
        i = i + l;
    end
end

% Q(h|h_old) = ln P(h) + integral p(x|data,h_old) ln p(data,x|h) dx
%            = ln P(h) + integral p(x|data,h_old) [ln p(data|x) + ln p(x|h)] dx
%           ~= ln P(h) + 1/L sum w(l) [ln p(data|x^l) + ln p(x^l|h)]
%
% (Bishop 2006, pg. 441)
% Last line is an importance sampling approximation of the integral
% using L samples x^1..x^L, with the importance weight of sample l given by
% w(l) = p(x^l|data,h_old) / q(x^l)
%
% TODO optimization: can ignore p(data|x) when maximizing Q w.r.t. h
%
function Q = computeQ(h_new, h_old, X, w, data, hparam, param, likfun)

lkajdflskjafl
    % initialize Q = ln P(h)
    Q = 0;
    i = 1;
    for k = 1:length(hparam)
        l = length(hparam(k).lb);
        Q = Q + hparam(k).logpdf(h(i:i+l-1));
        i = i + l;
    end

    % d
end
