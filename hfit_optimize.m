function results = hfit_optimize(likfun,hyparam,param,data,verbose)

    % Estimate hyperparameters (parameters for a group-level prior) by inverting
    % the generative model:
    %   group-level hyperparameters h ~ P(h), defined by hyparam.logpdf
    %   individual parameters       x ~ P(x|h), defined by param.hlogpdf
    %   data                     data ~ P(data|x), defined by likfun 
    %
    % Uses expectation maximization (EM) to find MAP P(h|data), with x serving as latent
    % variables (Bishop 2006). We approximate the integral over x using importance
    % sampling, with x's resampled every few iterations using Gibbs sampling, 
    % using Metropolis-Hastings to sample each component.
    %
    % USAGE: param = hfit_optimize(likfun,hyparam,param,data)
    %        results = mfit_optimize(likfun,param,data,[nstarts])
    %
    % INPUTS:
    %   likfun - likelihood function handle
    %   hyparam - [K x 1] hyperparameter structure
    %   param - [K x 1] parameter structure, with hlogpdf and hrnd fields
    %   data - [S x 1] data structure
    %   verbose (optional) - whether to print stuff
    %
    % OUTPUTS:
    %   param - the param structure array with .logpdf defined based on
    %           the MAP hyperparameters h. Can be passed in turn to mfit_optimize.
    %
    % Sam Gershman, July 2017

    if nargin < 5
        verbose = false;
    end
    
    % initialization
    tol = 1e-3; % tolerance: stop when improvements in Q are less than that
    maxiter = 20; % stop after that many iterations
    nsamples_x = 10; % how many samples of x to use to approximate EM integral
    init_samples_h = 100; % how many samples of h to use to initialize h_old
    batch_size = 50; % subsample once every batch_size samples (gibbs sampler param)
    burn_in = 10; % burn in first few samples when gibbs sampling

    K = length(param);
    assert(length(hyparam) == K, 'param and hyparam must have the same length');
    S = length(data);

    % pick a crude estimate of h_old 
    % compute a bunch of random h's and pick one with max P(h|D)
    disp('Initializing h_old with a crude estimate ...');
    logmargpost_old = -Inf;
    for i = 1:init_samples_h
        if verbose, disp(['  init iter ', num2str(i)]); end
        h = hyparam_rnd(hyparam, param);
        logp = loghypost(h, data, hyparam, param, likfun, nsamples_x);
        if logp > logmargpost_old
            h_old = h;
            logmargpost_old = logp;
            if verbose
                disp(['    new h_old = ', mat2str(h_old)]);
                disp(['    ln P(h|D) = ', num2str(logmargpost_old)]);
            end
        end
    end

    disp(['h_old initialized as ', mat2str(h_old)]);
    disp(['ln P(h|D) ~= ', num2str(logmargpost_old)]);
    
    % extract lower and upper bounds
    lb = [hyparam.lb];
    ub = [hyparam.ub];
    
    options = optimset('Display','off');
    warning off all
    
    % run expectation-maximization
    iter = 0;
    while iter < maxiter
        
        iter = iter + 1;
        disp(['.. iteration ',num2str(iter)]);

        %
        % E step: use importance sampling to approximate 
        % the integral over P(x|data,h_old)
        %

        %if ~exist('X', 'var') || eff < length(w) * 0.25
        %    [X, logq] = sample(h_old, likfun, hyparam, param, data, nsamples_x, batch_size, burn_in, verbose);
        %end

        % Draw samples from P(x|data,h_old)
        % TODO only do every so often
        [X, logq] = sample(h_old, likfun, hyparam, param, data, nsamples_x, batch_size, burn_in, verbose);

        [w, eff] = weights(X, data, h_old, hyparam, param, likfun, logq);

        %
        % M step: maximize Q(h,h_old)
        %

        f = @(h_new) -computeQ(h_new, h_old, X, w, data, hyparam, param, likfun, verbose);

        h0 = hyparam_rnd(hyparam, param);
    
        [h_new,nQ] = fmincon(f,h0,[],[],[],[],lb,ub,[],options);

        Q = -nQ;
        logmargpost_new = loghypost(h_new, data, hyparam, param, likfun, nsamples_x);

        % print stuff
        if verbose
            disp(['samples = ', mat2str(X)]);
            disp(['ln weights = ', mat2str(w)]);
        end
        disp(['h_new = ', mat2str(h_new)]);
        disp(['ln P(h_new|D) ~= ', num2str(logmargpost_new)]);
        if verbose
            disp('        ...vs...');
            disp(['    h_old = ', mat2str(h_old)]);
            disp(['    ln P(h_old|D) ~= ', num2str(logmargpost_old), ' (new one should be better)']);
        end

        h_old = h_new;
        logmargpost_old = logmargpost_new;
    end

    param = set_logpdf(hyparam, param, h_new);
end



% compute importance weights w for a given set of parameter samples
%
function [w, eff] = weights(X, data, h_old, hyparam, param, likfun, logq)
    % compute ln P(x|data,h_old) for samples (unnormalized)
    logp = logpost(X, data, h_old, hyparam, param, likfun);

    % compute importance weights
    % w(i) = p(i)/q(i) / sum(p(j)/q(j))   (Bishop 2006, p. 533)
    logw = logp - logq - logsumexp(logp - logq);
    w = exp(logw);

    % compute effective sample size
    eff = 1 / exp(logsumexp(logw * 2));
    disp(['effective sample size = ', num2str(eff), ' (vs. ', num2str(length(w)), ')']);
end


% Draw random samples of paramters from P(x|data,h) using Gibbs sampling (Bishop 2006, p. 543).
%
% OUTPUTS:
%   X = [nsamples x K] samples; each row is a set of parameters x
%   logq = [nsamples x 1] = ln q(x) = ln P(x|data,h) for each x (unnormalized); used to compute importance weights

%
function [X, logq] = sample(h_old, likfun, hyparam, param, data, nsamples, batch_size, burn_in, verbose)
    K = length(param);
    X = nan(nsamples, K);
    logq = nan(nsamples, 1);

    % initialize x0
    x_old = param_rnd(hyparam, param, h_old, true);

    % set .logpdf = P(x|h) for convenience
    param = set_logpdf(hyparam, param, h_old);

    % log of std of proposal distribution for each component
    ls = zeros(1,K);

    % Draw nsamples samples
    for n = 1:nsamples+burn_in 
        x_new = nan(1,K);

        if verbose, disp(['     sample ', num2str(n)]); end

        % Adaptive Metropolis-within-Gibbs for each component
        % (Roberts and Rosenthal, 2008)
        for k = 1:K
            % P(x_k|x_\k,data,h) is proportional to P(data|x) P(x_k|h)
            logpost_k = @(x_k) loglik([x_new(1:k-1) x_k x_old(k+1:end)], data, likfun) + param(k).logpdf(x_k);

            % proposals update by adding an increment ~ N(0,exp(ls(k))
            proprnd = @(x_old_k) mh_proprnd(x_old_k, param, k, ls);

            % draw batch_size samples of k-th component and take the last one only
            [x, accept] = mhsample(x_old(k), batch_size, 'logpdf', logpost_k, 'proprnd', proprnd, 'symmetric', true);
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

    X = X(burn_in+1:end,:);

    % compute q(x) = P(x|data,h) up to a proportionality constant
    logq = logpost(X, data, h_old, hyparam, param, likfun);
end


% Metropolis proposal function for x(k)_new given x(k)_old
%
function x_new_k = mh_proprnd(x_old_k, param, k, ls)
    while true
        x_new_k = normrnd(x_old_k, exp(ls(k)));
        if param(k).lb <= x_new_k && x_new_k <= param(k).ub
            % keep parameters within bounds
            break;
        end
    end
end


% ln P(x|h)
%
function logp = logprior(x, h, hyparam, param)
    logp = 0;
    i = 1;
    for k = 1:length(param)
        l = length(hyparam(k).lb);
        logp = logp + param(k).hlogpdf(x(k),h(i:i+l-1));
        i = i + l;
    end
    %disp(['  logprior ', num2str(x(1)), ',', num2str(x(2)), ' = ', num2str(logp)]);
end


% ln P(data|x) = sum ln P(data(s)|x)
% notice this is a single parameter vector and the likelihood
% is taken using data for all subjects
%
function logp = loglik(x, data, likfun)
    logp = 0;
    for s = 1:length(data)
        logp = logp + likfun(x,data(s));
    end
    %disp(['  loglik ', num2str(x(1)), ',', num2str(x(2)), ' = ', num2str(logp)]);
end


% ln P(x|data,h) for all samples (rows) x in X
% up to a proportionality constant.
%
% P(x|data,h) = P(data|x) P(x|h) / P(data|h)
%
function logp = logpost(X, data, h, hyparam, param, likfun)
    nsamples = size(X,1);
    for n = 1:nsamples
        x = X(n,:);
        logp(n) = loglik(x, data, likfun) + logprior(x, h, hyparam, param);
    end
end


% ln P(h)
%
function logp = loghyprior(h, hyparam)
    logp = 0;
    i = 1;
    for k = 1:length(hyparam)
        l = length(hyparam(k).lb);
        logp = logp + hyparam(k).logpdf(h(i:i+l-1));
        i = i + l;
    end
end
 

% ln P(D|h) approximation
%
% P(D|h) = integral P(D|x) P(x|h) dx
%       ~= 1/L sum P(D|x^l) P(x^l|h)
% where the L samples x^1..x^l are drawn from P(x|h)
%
function logp = loghylik(h, data, hyparam, param, likfun, nsamples)
    logp = [];
    for n = 1:nsamples
        x = param_rnd(hyparam, param, h, false);
        logpost = loglik(x, data, likfun) + logprior(x, h, hyparam, param);
        if ~isnan(logpost) && ~isinf(logpost)
            logp = [logp; logpost];
        end
    end
    assert(~isempty(logp), ['Looks like no good parameters x were found for the given h= ', mat2str(h)]);
    logp = logsumexp(logp) - log(nsamples);
end


% ln P(h|D) up to a proportionality constant (approximation)
%
function logp = loghypost(h, data, hyparam, param, likfun, nsamples)
    logp = loghylik(h, data, hyparam, param, likfun, nsamples) + loghyprior(h, hyparam);
end


% set .logpdf i.e. P(x|h) for each param based on given hyperparameters h
%
function param = set_logpdf(hyparam, param, h)
    i = 1;
    for k = 1:length(param)
        l = length(hyparam(k).lb);
        param(k).logpdf = @(x) param(k).hlogpdf(x,h(i:i+l-1));
        i = i + l;
    end
end


% random draw from P(h)
%
function h = hyparam_rnd(hyparam, param)
    h = [];
    for k = 1:length(hyparam)
        while true
            h_k = hyparam(k).rnd();
            x_k = param(k).hrnd(h_k);
            logp = param(k).hlogpdf(x_k, h_k);
            if all(hyparam(k).lb <= h_k) && all(h_k <= hyparam(k).ub) && ~isinf(logp) && ~isnan(logp)
                % keep hyperparameters within bounds and do a sanity check
                % to make sure they don't generate invalid parameters
                h = [h h_k];
                break;
            end
        end
    end
end


% random draw from P(x|h)
%
function x = param_rnd(hyparam, param, h, respect_bounds)
    i = 1;
    for k = 1:length(param)
        l = length(hyparam(k).lb);
        while true
            x(k) = param(k).hrnd(h(i:i+l-1));
            if ~respect_bounds || (param(k).lb <= x(k) && x(k) <= param(k).ub)
                % keep parameters within bounds
                break;
            end
        end
        i = i + l;
    end
end



% Q(h|h_old) = ln P(h) + integral P(x|data,h_old) ln P(data,x|h) dx
%            = ln P(h) + integral P(x|data,h_old) [ln P(data|x) + ln P(x|h)] dx
%           ~= ln P(h) + sum w(l) [ln P(data|x^l) + ln P(x^l|h)]
% (Bishop 2006, p. 441 and p. 536)
% Last line is an importance sampling approximation of the integral
% using L samples x^1..x^L, with the importance weight of sample l given by
% w(l) = P(x^l|data,h_old) / q(x^l) / sum(...)      (Bishop 2006, p. 533)
%
% TODO optimization: can ignore P(data|x) when maximizing Q w.r.t. h
%
function Q = computeQ(h_new, h_old, X, w, data, hyparam, param, likfun, verbose)
    nsamples = size(X,1);

    % integral P(x|data,h_old) ln P(data,x|h) dx
    % approximated using importance sampling
    Q = 0;
    for n = 1:nsamples
        x = X(n,:);
        Q = Q + w(n) * (loglik(x, data, likfun) + logprior(x, h_new, hyparam, param));
    end 

    % ln P(h)
    Q = Q + loghyprior(h_new, hyparam); 

    if verbose
        disp(['   computeQ: h_new = ', mat2str(h_new)]);
        disp(['       P(h) = ', num2str(loghyprior(h_new, hyparam)), ' Q = ', num2str(Q)]);
    end
end
