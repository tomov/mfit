function results = hfit_optimize(likfun,hparam,param,data)

    % Estimate hyperparameters (parameters for a group-level prior) by inverting
    % the generative model:
    %   group-level hyperparameters h ~ P(h), defined by hparam.logpdf
    %   individual parameters       x ~ P(x|h), defined by param.hlogpdf
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
        [X, logq] = sample(h_old, likfun, hparam, param, data, nsamples);

        % compute ln P(x|data,h_old) for samples (unnormalized)
        logp = logpost(X, data, h_old, hparam, param, likfun);

        % compute importance weights
        % w(i) = p(i)/q(i) / sum(p(j)/q(j))   (Bishop 2006, p. 533)
        logw = logp - logq - logsumexp(logp - logq);
        w = exp(logw);

        disp(X);
        disp(w);

        f = @(h_new) -computeQ(h_new, h_old, X, w, data, hparam, param, likfun);

        h0 = [];
        for k = 1:K
            h0 = [h0 hparam(k).rnd()];
        end
    
        [h_new,nQ] = fmincon(f,h0,[],[],[],[],lb,ub,[],options);
        Q = -nQ;
            
        h_old = h_new;

        disp('NEW h!');
        disp(h_new);
        disp(loghypost(h_new, data, hparam, param, likfun));
    end

    param = set_logpdf(hparam, param, h_new);
end



% Draw random samples of paramters from P(x|data,h) using Gibbs sampling (Bishop 2006, p. 543).
%
% OUTPUTS:
%   X = [nsamples x K] samples; each row is a set of parameters x
%   logq = [nsamples x 1] = ln q(x) = ln P(x|data,h) for each x (unnormalized); used to compute importance weights

%
function [X, logq] = sample(h_old, likfun, hparam, param, data, nsamples)
    K = length(param);
    X = nan(nsamples, K);
    logq = nan(nsamples, 1);
    batch_size = 50; % subsample once every batch_size samples
    burn_in = 10; % burn in first few samples

    % initialize x0
    x_old = param_rnd(hparam, param, h_old);

    % set .logpdf = P(x|h) for convenience
    param = set_logpdf(hparam, param, h_old);

    % log of std of proposal distribution for each component
    ls = zeros(1,K);

    % Draw nsamples samples
    for n = 1:nsamples+burn_in 
        x_new = nan(1,K);

        disp(['     sample ', num2str(n)]);

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
    logq = logpost(X, data, h_old, hparam, param, likfun);
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
function logp = logprior(x, h, hparam, param)
    logp = 0;
    i = 1;
    for k = 1:length(param)
        l = length(hparam(k).lb);
        logp = logp + param(k).hlogpdf(x(k),h(i:i+l-1));
        i = i + l;
    end
    h
    disp(['  logprior ', num2str(x(1)), ',', num2str(x(2)), ' = ', num2str(logp)]);
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
    disp(['  loglik ', num2str(x(1)), ',', num2str(x(2)), ' = ', num2str(logp)]);
end


% ln P(x|data,h) for all samples (rows) x in X
% up to a proportionality constant.
%
% P(x|data,h) = P(data|x) P(x|h) / P(data|h)
%
function logp = logpost(X, data, h, hparam, param, likfun)
    nsamples = size(X,1);
    for n = 1:nsamples
        x = X(n,:);
        logp(n) = loglik(x, data, likfun) + logprior(x, h, hparam, param);
    end
end


% ln P(h)
%
function logp = loghyprior(h, hparam)
    logp = 0;
    i = 1;
    for k = 1:length(hparam)
        l = length(hparam(k).lb);
        logp = logp + hparam(k).logpdf(h(i:i+l-1));
        i = i + l;
    end
end
 

% ln P(D|h) approximation
%
% P(D|h) = integral P(D|x) P(x|h) dx
%       ~= 1/L sum P(D|x^l) P(x^l|h)
% where the L samples x^1..x^l are drawn from P(x|h)
%
function logp = loghylik(h, data, hparam, param, likfun)
    nsamples = 10;
    logp = [];
    disp('----------------sheeeeeeeeeeeeeeeeit');
    for n = 1:nsamples
        x = param_rnd(hparam, param, h);
        x
        logp = [logp; loglik(x, data, likfun) + logprior(x, h, hparam, param)];
    end
    logp = logsumexp(logp) - log(nsamples);
end


% ln P(h|D) up to a proportionality constant
%
function logp = loghypost(h, data, hparam, param, likfun)
    logp = loghylik(h, data, hparam, param, likfun) + loghyprior(h, hparam);
end


% random draw from P(x|h)
%
function x = param_rnd(hparam, param, h)
    i = 1;
    for k = 1:length(param)
        l = length(hparam(k).lb);
        x(k) = param(k).hrnd(h(i:i+l-1));
        i = i + l;
    end
end



% set .logpdf i.e. P(x|h) for each param based on given hyperparameters h
%
function param = set_logpdf(hparam, param, h)
    i = 1;
    for k = 1:length(param)
        l = length(hparam(k).lb);
        param(k).logpdf = @(x) param(k).hlogpdf(x,h(i:i+l-1));
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
function Q = computeQ(h_new, h_old, X, w, data, hparam, param, likfun)
    nsamples = size(X,1);

    % integral P(x|data,h_old) ln P(data,x|h) dx
    % approximated using importance sampling
    Q = 0;
    for n = 1:nsamples
        x = X(n,:);
        Q = Q + w(n) * (loglik(x, data, likfun) + logprior(x, h_new, hparam, param));
    end 

    % ln P(h)
    Q = Q + loghyprior(h_new, hparam); 

    disp('computeQ');
    disp(h_new);
    disp(loghyprior(h_new, hparam));
    disp(Q);

end
