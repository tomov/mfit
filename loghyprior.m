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
 

