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

        % compute P(x|data,h_old) for samples
        logp = logpost(X, data, h_old, hparam, param, likfun);
        logp = logp - logsumexp(logp); % normalize by P(data,h_old) (approximately)

        % compute importance weights
        w = exp(logp - logq);

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
    end

    param = set_logpdf(hparam, param, h_new);
end



% Draw random samples of paramters from P(x|data,h) using Gibbs sampling (Bishop 2006, p. 543).
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
        % See Roberts and Rosenthal (2008): Examples of Adaptive MCMC
        for k = 1:K
            % P(x_k|x_\k,data,h) is proportional to P(data|x) P(x_k|h)
            logpost_k = @(x_k) loglik([x_new(1:k-1) x_k x_old(k+1:end)], data, likfun) + param(k).logpdf(x_k);

            % proposals update by adding an increment ~ N(0,exp(ls(k))
            proprnd = @(x_old) normrnd(x_old, exp(ls(k)));
            proppdf = @(x_new,x_old) normpdf(x_new, x_old, exp(ls(k)));

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

    % Compute proportionality constant P(data,h) and normalize q(x)'s by it.
    % This is important since the normalization constant is different for different
    % values of h, and thus ignoring it would screw up the importance weights.
    % We approximate the integral with a sum
    C = logsumexp(logq);
    logq = logq - C;
end



% ln P(h)
%
function logp = loghyperprior(h, hparam)
    logp = 0;
    i = 1;
    for k = 1:length(hparam)
        l = length(hparam(k).lb);
        logp = logp + hparam(k).logpdf(h(i:i+l-1));
        i = i + l;
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


% Calculate ln P(x|data,h) for all samples (rows) x in X
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



% random draw from P(x|h)
%
function x = param_rnd(hparam, param, h)
    i = 1;
    for k = 1:length(param)
        l = length(hparam(k).lb);
        x(k) = param(k).hrnd(h(i:i+l-1));
        % rejection sampling -- not needed TODO rm
        %while true
        %    x(k) = param(k).hrnd(h(i:i+l-1));
        %    if param(k).lb <= x(k) && x(k) >= param(k).ub
        %        break;
        %    end
        %end
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
%           ~= ln P(h) + 1/L sum w(l) [ln P(data|x^l) + ln P(x^l|h)]
% (Bishop 2006, p. 441 and p. 536)
% Last line is an importance sampling approximation of the integral
% using L samples x^1..x^L, with the importance weight of sample l given by
% w(l) = P(x^l|data,h_old) / q(x^l)
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
    Q = Q / nsamples;

    % ln P(h)
    Q = Q + loghyperprior(h_new, hparam); 

    disp('computeQ');
    disp(h_new);
    disp(loghyperprior(h_new, hparam));
    disp(Q);

end
